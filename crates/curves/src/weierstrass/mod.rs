use generic_array::GenericArray;
use num::{BigUint, Zero};
use serde::{Deserialize, Serialize};

use super::CurveType;
use crate::{
    params::{FieldParameters, NumLimbs, NumWords},
    utils::biguint_to_bits_le,
    AffinePoint, EllipticCurve, EllipticCurveParameters,
};
use alloy_primitives::{Uint, U256, U512};

#[cfg(feature = "bigint-rug")]
use crate::utils::{biguint_to_rug, rug_to_biguint};

pub mod bls12_381;
pub mod bn254;
pub mod secp256k1;

/// Parameters that specify a short Weierstrass curve : y^2 = x^3 + ax + b.
pub trait WeierstrassParameters: EllipticCurveParameters {
    const A: GenericArray<u8, <Self::BaseField as NumLimbs>::Limbs>;
    const B: GenericArray<u8, <Self::BaseField as NumLimbs>::Limbs>;

    fn generator() -> (BigUint, BigUint);

    fn prime_group_order() -> BigUint;

    fn a_int() -> BigUint {
        let mut modulus = BigUint::zero();
        for (i, limb) in Self::A.iter().enumerate() {
            modulus += BigUint::from(*limb) << (8 * i);
        }
        modulus
    }

    fn b_int() -> BigUint {
        let mut modulus = BigUint::zero();
        for (i, limb) in Self::B.iter().enumerate() {
            modulus += BigUint::from(*limb) << (8 * i);
        }
        modulus
    }

    fn nb_scalar_bits() -> usize {
        Self::BaseField::NB_LIMBS * 16
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwCurve<E>(pub E);

impl<E: WeierstrassParameters> WeierstrassParameters for SwCurve<E> {
    const A: GenericArray<u8, <Self::BaseField as NumLimbs>::Limbs> = E::A;
    const B: GenericArray<u8, <Self::BaseField as NumLimbs>::Limbs> = E::B;

    fn a_int() -> BigUint {
        E::a_int()
    }

    fn b_int() -> BigUint {
        E::b_int()
    }

    fn generator() -> (BigUint, BigUint) {
        E::generator()
    }

    fn nb_scalar_bits() -> usize {
        E::nb_scalar_bits()
    }

    fn prime_group_order() -> BigUint {
        E::prime_group_order()
    }
}

impl<E: WeierstrassParameters> EllipticCurveParameters for SwCurve<E> {
    type BaseField = E::BaseField;

    const CURVE_TYPE: CurveType = E::CURVE_TYPE;
}

impl<E: WeierstrassParameters> EllipticCurve for SwCurve<E> {
    const NB_LIMBS: usize = Self::BaseField::NB_LIMBS;
    const NB_WITNESS_LIMBS: usize = Self::BaseField::NB_WITNESS_LIMBS;

    fn ec_add(p: &AffinePoint<Self>, q: &AffinePoint<Self>) -> AffinePoint<Self> {
        p.sw_add(q)
    }

    fn ec_double(p: &AffinePoint<Self>) -> AffinePoint<Self> {
        p.sw_double()
    }

    fn ec_generator() -> AffinePoint<Self> {
        let (x, y) = E::generator();
        AffinePoint::new(x, y)
    }

    fn ec_neutral() -> Option<AffinePoint<Self>> {
        None
    }

    fn ec_neg(p: &AffinePoint<Self>) -> AffinePoint<Self> {
        let modulus = E::BaseField::modulus();
        AffinePoint::new(p.x.clone(), modulus - &p.y)
    }
}

impl<E: WeierstrassParameters> SwCurve<E> {
    pub fn generator() -> AffinePoint<SwCurve<E>> {
        let (x, y) = E::generator();

        AffinePoint::new(x, y)
    }

    pub fn a_int() -> BigUint {
        E::a_int()
    }

    pub fn b_int() -> BigUint {
        E::b_int()
    }
}

impl<E: WeierstrassParameters> AffinePoint<SwCurve<E>> {
    pub fn sw_scalar_mul(&self, scalar: &BigUint) -> Self {
        let mut result: Option<AffinePoint<SwCurve<E>>> = None;
        let mut temp = self.clone();
        let bits = biguint_to_bits_le(scalar, E::nb_scalar_bits());
        for bit in bits {
            if bit {
                result = result.map(|r| r.sw_add(&temp)).or(Some(temp.clone()));
            }
            temp = temp.sw_double();
        }
        result.unwrap()
    }
}

pub fn biguint_to_dashu(integer: &BigUint) -> dashu::integer::UBig {
    dashu::integer::UBig::from_le_bytes(integer.to_bytes_le().as_slice())
}

pub fn dashu_to_biguint(integer: &dashu::integer::UBig) -> BigUint {
    BigUint::from_bytes_le(&integer.to_le_bytes())
}

pub fn dashu_modpow(
    base: &dashu::integer::UBig,
    exponent: &dashu::integer::UBig,
    modulus: &dashu::integer::UBig,
) -> dashu::integer::UBig {
    if modulus == &dashu::integer::UBig::from(1u32) {
        return dashu::integer::UBig::from(0u32);
    }

    assert!(base.to_le_bytes().len() <= 32, "{:x?}", base.to_le_bytes());
    assert!(exponent.to_le_bytes().len() <= 32);
    assert!(modulus.to_le_bytes().len() <= 32);

    let u256_res = {
        let modulus = U256::from_le_slice(&modulus.to_le_bytes());

        let base = U256::from_le_slice(&base.to_le_bytes());
        let exp = U256::from_le_slice(&exponent.to_le_bytes()) % modulus;

        let result = base.pow_mod(exp, modulus);

        dashu::integer::UBig::from_le_bytes(result.as_le_slice())
    };

    return u256_res;

    let mut result = dashu::integer::UBig::from(1u32);
    let mut base = base.clone() % modulus;
    let mut exp = exponent.clone();

    let mut i = 0;

    while exp > dashu::integer::UBig::from(0u32) {
        println!("{i}: {result}");

        if &exp % dashu::integer::UBig::from(2u32) == dashu::integer::UBig::from(1u32) {
            result = (result * &base) % modulus;
        }
        exp >>= 1;
        base = (&base * &base) % modulus;

        i += 1;
    }

    assert_eq!(result, u256_res, "{result} != {u256_res}");

    result
}

impl<E: WeierstrassParameters> AffinePoint<SwCurve<E>> {
    pub fn sw_add(&self, other: &AffinePoint<SwCurve<E>>) -> AffinePoint<SwCurve<E>> {
        if self.x == other.x && self.y == other.y {
            panic!("Error: Points are the same. Use sw_double instead.");
        }

        cfg_if::cfg_if! {
            if #[cfg(feature = "bigint-rug")] {
                self.sw_add_rug(other)
            } else {

                let p = U256::from_le_slice(E::BaseField::MODULUS);
                let self_x = U256::from_le_slice(self.x.to_bytes_le().as_slice());
                let self_y = U256::from_le_slice(self.y.to_bytes_le().as_slice());
                let other_x = U256::from_le_slice(other.x.to_bytes_le().as_slice());
                let other_y = U256::from_le_slice(other.y.to_bytes_le().as_slice());


                let slope_num = other_y.add_mod(p - self_y, p);
                let slope_denom = other_x.add_mod(p - self_x, p);
                let slope = slope_num.mul_mod(slope_denom.inv_mod(p).unwrap(), p);

                let x_3n = slope.mul_mod(slope, p).add_mod(p - self_x, p).add_mod(p - other_x, p);
                let y_3n = self_x.add_mod(p - x_3n, p).mul_mod(slope, p).add_mod(p - self_y, p);

                return AffinePoint::new(
                    BigUint::from_bytes_le(x_3n.as_le_slice()),
                    BigUint::from_bytes_le(y_3n.as_le_slice())
                );




                let p = biguint_to_dashu(&E::BaseField::modulus());
                let self_x = biguint_to_dashu(&self.x);
                let self_y = biguint_to_dashu(&self.y);

                let other_x = biguint_to_dashu(&other.x);
                let other_y = biguint_to_dashu(&other.y);

                let slope_numerator = (&p + &other_y - &self_y) % &p;
                let slope_denominator = (&p + &other_x - &self_x) % &p;
                let slope_denom_inverse =
                    dashu_modpow(&slope_denominator, &(&p - &dashu::integer::UBig::from(2u32)), &p);
                let slope = (slope_numerator * &slope_denom_inverse) % &p;

                let x_3n = (&slope * &slope + &p + &p - &self_x - &other_x) % &p;
                let y_3n = (&slope * &(&p + &self_x - &x_3n) + &p - &self_y) % &p;

                AffinePoint::new(dashu_to_biguint(&x_3n), dashu_to_biguint(&y_3n))
            }
        }
    }

    fn sw_double_generic<const BITS: usize, const LIMBS: usize>(&self) -> AffinePoint<SwCurve<E>> {
        let p = Uint::<BITS, LIMBS>::from_le_slice(E::BaseField::MODULUS);
        let a = Uint::<BITS, LIMBS>::from_le_slice(E::a_int().to_bytes_le().as_slice());
        let x = Uint::<BITS, LIMBS>::from_le_slice(self.x.to_bytes_le().as_slice());
        let y = Uint::<BITS, LIMBS>::from_le_slice(self.y.to_bytes_le().as_slice());

        let slope_num = x
            .mul_mod(x, p)
            .mul_mod(Uint::<BITS, LIMBS>::from(3), p)
            .add_mod(a, p);
        let slope_denom = y.mul_mod(Uint::<BITS, LIMBS>::from(2), p);
        let slop_denom_inv = slope_denom.inv_mod(p).unwrap();
        let slope = slope_num.mul_mod(slop_denom_inv, p);

        let x_3n = slope.mul_mod(slope, p).add_mod(p - x, p).add_mod(p - x, p);
        let y_3n = x.add_mod(p - x_3n, p).mul_mod(slope, p).add_mod(p - y, p);

        return AffinePoint::new(
            BigUint::from_bytes_le(x_3n.as_le_slice()),
            BigUint::from_bytes_le(y_3n.as_le_slice()),
        );
    }

    pub fn sw_double(&self) -> AffinePoint<SwCurve<E>> {
        cfg_if::cfg_if! {
            if #[cfg(feature = "bigint-rug")] {
                self.sw_double_rug()
            } else {
                if E::BaseField::MODULUS.len() == 32 {
                    self.sw_double_generic::<256, 4>()
                } else {
                    assert!(E::BaseField::MODULUS.len() <= 64);
                    self.sw_double_generic::<512, 8>()
                }
            }
        }
    }

    #[cfg(feature = "bigint-rug")]
    pub fn sw_add_rug(&self, other: &AffinePoint<SwCurve<E>>) -> AffinePoint<SwCurve<E>> {
        use rug::Complete;
        let p = biguint_to_rug(&E::BaseField::modulus());
        let self_x = biguint_to_rug(&self.x);
        let self_y = biguint_to_rug(&self.y);
        let other_x = biguint_to_rug(&other.x);
        let other_y = biguint_to_rug(&other.y);

        let slope_numerator = ((&p + &other_y).complete() - &self_y) % &p;
        let slope_denominator = ((&p + &other_x).complete() - &self_x) % &p;
        let slope_denom_inverse = slope_denominator
            .pow_mod_ref(&(&p - &rug::Integer::from(2u32)).complete(), &p)
            .unwrap()
            .complete();
        let slope = (slope_numerator * &slope_denom_inverse) % &p;

        let x_3n = ((&slope * &slope + &p).complete() + &p - &self_x - &other_x) % &p;
        let y_3n = ((&slope * &((&p + &self_x).complete() - &x_3n) + &p).complete() - &self_y) % &p;

        AffinePoint::new(rug_to_biguint(&x_3n), rug_to_biguint(&y_3n))
    }

    #[cfg(feature = "bigint-rug")]
    pub fn sw_double_rug(&self) -> AffinePoint<SwCurve<E>> {
        use rug::Complete;
        let p = biguint_to_rug(&E::BaseField::modulus());
        let a = biguint_to_rug(&E::a_int());

        let self_x = biguint_to_rug(&self.x);
        let self_y = biguint_to_rug(&self.y);

        let slope_numerator = (&a + &(&self_x * &self_x).complete() * 3u32).complete() % &p;

        let slope_denominator = (&self_y * 2u32).complete() % &p;
        let slope_denom_inverse = slope_denominator
            .pow_mod_ref(&(&p - &rug::Integer::from(2u32)).complete(), &p)
            .unwrap()
            .complete();

        let slope = (slope_numerator * &slope_denom_inverse) % &p;

        let x_3n = ((&slope * &slope + &p).complete() + ((&p - &self_x).complete() - &self_x)) % &p;

        let y_3n = ((&slope * &((&p + &self_x).complete() - &x_3n) + &p).complete() - &self_y) % &p;

        AffinePoint::new(rug_to_biguint(&x_3n), rug_to_biguint(&y_3n))
    }
}

#[derive(Debug)]
pub enum FieldType {
    Bls12381,
    Bn254,
}

pub trait FpOpField: FieldParameters + NumWords {
    const FIELD_TYPE: FieldType;
}

#[cfg(test)]
mod tests {

    use num::bigint::RandBigInt;
    use rand::thread_rng;

    use super::bn254;

    #[test]
    fn test_weierstrass_biguint_scalar_mul() {
        type E = bn254::Bn254;
        let base = E::generator();

        let mut rng = thread_rng();
        for _ in 0..10 {
            let x = rng.gen_biguint(24);
            let y = rng.gen_biguint(25);

            let x_base = base.sw_scalar_mul(&x);
            let y_x_base = x_base.sw_scalar_mul(&y);
            let xy = &x * &y;
            let xy_base = base.sw_scalar_mul(&xy);
            assert_eq!(y_x_base, xy_base);
        }
    }
}