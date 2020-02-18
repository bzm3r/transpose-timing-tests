extern crate rand;
use rand::prelude::*;

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
        for i in 0..32 {
            for j in 0..32 {
                if !(self.0[i][j] == other.0[i][j]) {
                    return false;
                }
            }
        }
        true
    }

    fn row_to_u32(&self, i: usize) -> u32 {
        let row = self.0[i];
        let mut r: u32 = 0;
        for j in 0u32..32u32 {
            if row[j] {
                r = (1u32 << j) | r;
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
            let mut bm= [[false; u32]; 32];
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
