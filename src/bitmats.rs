extern crate rand;

use rand::prelude::*;
use std::fmt;
use std::fmt::Write;

pub struct BitMatrix([[bool; 32]; 32]);

impl BitMatrix {
    pub fn new_random() -> BitMatrix {
        let mut bm: [[bool; 32]; 32] = [[false; 32]; 32];
        for i in 0..32 {
            for j in 0..32 {
                bm[i][j] = random();
            }
        }
        BitMatrix(bm)
    }

    pub fn new_diagonal() -> BitMatrix {
        let mut bm: [[bool; 32]; 32] = [[false; 32]; 32];
        for i in 0..32 {
            bm[i][i] = true;
        }
        BitMatrix(bm)
    }

    pub fn new_corner() -> BitMatrix {
        let mut bm: [[bool; 32]; 32] = [[false; 32]; 32];
        bm[0][31] = true;
        BitMatrix(bm)
    }

    pub fn transpose(&self) -> BitMatrix {
        let mut bm: [[bool; 32]; 32] = [[false; 32]; 32];
        for i in 0..32 {
            for j in 0..32 {
                bm[i][j] = self.0[j][i];
            }
        }
        BitMatrix(bm)
    }

    pub fn identical_to(&self, other: &BitMatrix) -> bool {
        self.as_u32s()
            .iter()
            .zip(other.as_u32s().iter())
            .all(|(x, y)| x == y)
    }

    fn row_to_u32(&self, i: usize) -> u32 {
        let row = self.0[i];
        let mut r: u32 = 0;
        for j in 0..32 {
            if row[j] {
                r = (1 << j as u32) | r;
            }
        }
        r
    }

    pub fn as_u32s(&self) -> [u32; 32] {
        let mut r: [u32; 32] = [0; 32];
        for i in 0..32 {
            r[i] = self.row_to_u32(i);
        }
        r
    }

    pub fn from_u32s(input: &[u32]) -> Result<BitMatrix, ()> {
        if input.len() != 32 {
            Err(())
        } else {
            let mut bm = [[false; 32]; 32];
            for i in 0..32 {
                let row = input[i];
                for j in 0..32 {
                    bm[i][j] = (row & (1u32 << j as u32)) > 0;
                }
            }
            Ok(BitMatrix(bm))
        }
    }
}

impl fmt::Display for BitMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut string_repr = String::new();
        for (i, u) in self.as_u32s().iter().enumerate() {
            if i == 0 {
                write!(string_repr, "[{}, ", u).unwrap();
            } else if i == 31 {
                write!(string_repr, "{}]", u).unwrap();
            } else {
                write!(string_repr, "{}, ", u).unwrap();
            }
        }
        write!(f, "{}", string_repr)
    }
}
