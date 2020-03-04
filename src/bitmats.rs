extern crate rand;

use rand::prelude::*;
use std::fmt;
use std::fmt::Write;

#[derive(Clone, Copy)]
pub enum Interpretation {
    B8,
    B32,
}

pub struct BitMatrix {
    raw: [[bool; 32]; 32],
    interpretation: Interpretation,
}

impl BitMatrix {
    pub fn new_random(interpretation: Interpretation) -> BitMatrix {
        let mut raw: [[bool; 32]; 32] = [[false; 32]; 32];
        for i in 0..32 {
            for j in 0..32 {
                raw[i][j] = random();
            }
        }
        BitMatrix {
            raw,
            interpretation,
        }
    }

    pub fn new_cross_diagonal(interpretation: Interpretation) -> BitMatrix {
        let mut raw: [[bool; 32]; 32] = [[false; 32]; 32];

        match &interpretation {
            Interpretation::B8 => {
                for i in 0..32 {
                    for j in 0..32 {
                        if ((31 - i) % 8) == (j % 8) {
                            raw[i][j] = true;
                        }
                    }
                }
            },
            Interpretation::B32 => {
                for i in 0..32 {
                    raw[(31 - i)][i] = true;
                }
            }
        }

        BitMatrix {
            raw,
            interpretation,
        }
    }

    pub fn new_corner(interpretation: Interpretation) -> BitMatrix {
        let mut raw: [[bool; 32]; 32] = [[false; 32]; 32];

        match &interpretation {
            Interpretation::B8 => {
                for i in 0..32 {
                    for j in 0..32 {
                        if (i % 8) == 7 && (j % 8 == 0) {
                            raw[i][j] = true;
                        }
                    }
                }
            },
            Interpretation::B32 => {
                raw[31][0] = true;
            }
        };

        BitMatrix {
            raw,
            interpretation,
        }
    }

    pub fn new_constant(c: u32, interpretation: Interpretation) -> BitMatrix {
        let c = match  &interpretation {
            Interpretation::B8 => {
                assert!(c < 256);
                let mut mirrored_c: u32 = 0;
                for i in 0..4 {
                    mirrored_c = mirrored_c | (c << i*8);
                }
                mirrored_c
            }
            Interpretation::B32 => {
                c
            }
        };

        let ibm = [c; 32];
        BitMatrix::from_u32s(&ibm, interpretation).unwrap()
    }

    pub fn transpose(&self) -> BitMatrix {
        let mut raw: [[bool; 32]; 32] = [[false; 32]; 32];

        match self.interpretation {
            Interpretation::B8 => {
                for i in 0..32 {
                    for j in 0..32 {
                        let (e0x, e0y) = (i - (i % 8), j - (j % 8));
                        let (u, v) = (i - e0x, j - e0y);
                        let (ti, tj) = (e0x + v, e0y + u);
                        raw[i][j] = self.raw[ti][tj];
                        raw[ti][tj] = self.raw[i][j];
                    }
                }
            },
            Interpretation::B32 => {
                for i in 0..32 {
                    for j in 0..32 {
                        raw[(31 - i)][j] = self.raw[j][(31 - i)];
                    }
                }
            },
        }

        BitMatrix {
            raw,
            interpretation: self.interpretation,
        }
    }

    pub fn identical_to(&self, other: &BitMatrix) -> bool {
        self.as_u32s()
            .iter()
            .zip(other.as_u32s().iter())
            .all(|(x, y)| x == y)
    }

    fn row_to_u32(&self, i: usize) -> u32 {
        let row = self.raw[i];
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

    pub fn from_u32s(input: &[u32], interpretation: Interpretation) -> Result<BitMatrix, ()> {
        if input.len() != 32 {
            Err(())
        } else {
            let mut raw = [[false; 32]; 32];
            for i in 0..32 {
                let row = input[i];
                for j in 0..32 {
                    raw[i][j] = (row & (1u32 << j as u32)) > 0;
                }
            }
            Ok(BitMatrix {
                raw,
                interpretation,
            })
        }
    }
}

impl fmt::Display for BitMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut string_repr = String::new();
        match self.interpretation {
            Interpretation::B8 => {
                let raw_u32s = self.as_u32s();

                for i in 0..4 {
                    for j in 0..4 {
                        for u in 0..8 {
                            let x = (raw_u32s[j*8 + u] >> i*8) & 255;
                            if u == 0 {
                                write!(string_repr, "[{}, ", x).unwrap();
                            } else if u == 7 {
                                write!(string_repr, "{}]", x).unwrap();
                            } else {
                                write!(string_repr, "{}, ", x).unwrap();
                            }
                        }
                        if i*j < 15 {
                            write!(string_repr, ",\n").unwrap();
                        }
                    }
                }
            },
            Interpretation::B32 => {
                for (i, u) in self.as_u32s().iter().enumerate() {
                    if i == 0 {
                        write!(string_repr, "[{}, ", u).unwrap();
                    } else if i == 31 {
                        write!(string_repr, "{}]", u).unwrap();
                    } else {
                        write!(string_repr, "{}, ", u).unwrap();
                    }
                }
            }
        }

        write!(f, "{}", string_repr)
    }
}
