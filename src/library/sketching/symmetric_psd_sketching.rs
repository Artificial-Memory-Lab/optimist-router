use linfa_linalg::eigh::{EigSort, Eigh};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use std::cmp::min;

#[typetag::serde(tag = "psd-sketching")]
pub trait SymmetricPSDSketch: Sync + Send {
    /// Returns `u.t() S u` if S is the sketched matrix.
    fn dot(&self, u: ArrayView1<f32>) -> f32;

    /// Returns the size of the sketch in terms.
    fn size(&self) -> usize;
}

pub trait SymmetricPSDSketcher: Sync + Send {
    /// Sketches a symmetric positive semidefinite matrix.
    fn sketch(&self, matrix: ArrayView2<f32>) -> Box<dyn SymmetricPSDSketch>;
}

#[derive(Serialize, Deserialize)]
pub struct IdentitySketch {
    sketch: Array2<f32>,
}

#[typetag::serde]
impl SymmetricPSDSketch for IdentitySketch {
    fn dot(&self, u: ArrayView1<f32>) -> f32 {
        u.dot(&self.sketch.dot(&u))
    }

    fn size(&self) -> usize {
        self.sketch.ncols() * self.sketch.nrows()
    }
}

#[derive(Default)]
pub struct IdentitySketcher {}

impl SymmetricPSDSketcher for IdentitySketcher {
    fn sketch(&self, matrix: ArrayView2<f32>) -> Box<dyn SymmetricPSDSketch> {
        Box::new(IdentitySketch {
            sketch: matrix.to_owned(),
        })
    }
}

#[derive(Serialize, Deserialize)]
pub struct LowRankSketch {
    /// Stores the top eigen values and vectors of D^{-1/2} R D^{-1/2} where D is the diagonal
    /// and R=C-D is the residual of the matrix C.
    eigen_values: Array1<f32>,

    diag: Array1<f32>,

    /// Stores the square root of the diagonal of the matrix (D^{1/2} * U)
    eigen_vectors_diag_scaled: Array2<f32>,
}

#[typetag::serde]
impl SymmetricPSDSketch for LowRankSketch {
    fn dot(&self, u: ArrayView1<f32>) -> f32 {
        // First compute u.t() D^{1/2} Q S Q.t() D^{1/2} u
        // where Q contains the top eigenvectors and S the corresponding eigenvalues.
        let residual_score = self
            .eigen_vectors_diag_scaled
            .axis_iter(Axis(0))
            .zip(self.eigen_values.iter())
            .map(|(r, l)| {
                let x = r.dot(&u);
                x * x * l
            })
            .sum::<f32>();

        let u = &u * &u;
        let diag_score = self.diag.dot(&u);

        // Now add the two scores.
        diag_score + residual_score
    }

    fn size(&self) -> usize {
        self.eigen_vectors_diag_scaled.len() + self.eigen_values.len() + self.diag.len()
    }
}

pub struct LowRankApproximation {
    rank: usize,
}

impl LowRankApproximation {
    pub fn new(rank: usize) -> LowRankApproximation {
        LowRankApproximation { rank }
    }
}

impl SymmetricPSDSketcher for LowRankApproximation {
    /// Suppose we wish to sketch a PSD matrix C via low-rank approximation.
    /// We first extract its diagonal D, and form the residual matrix R = C - D.
    ///
    /// Knowing that C = D + R, we can write it as C = D^{1/2} (I + D^{-1/2} R D^{-1/2}) D^{1/2}.
    /// Note that the middle term is PSD.
    ///
    /// We can therefore store D^{1/2}, as well as the top eigenvalues and corresponding
    /// eigenvectors of D^{-1/2} R D^{-1/2} to approximate C.
    fn sketch(&self, matrix: ArrayView2<f32>) -> Box<dyn SymmetricPSDSketch> {
        let rank = min(self.rank, matrix.ncols());

        let diag = matrix.diag().to_owned();

        let mut diag_sqrt = diag.clone();
        diag_sqrt.mapv_inplace(|v| v.sqrt());

        let mut diag_sqrt_inv = diag.clone();
        diag_sqrt_inv.mapv_inplace(|v| if v != 0. { 1_f32 / v.sqrt() } else { 0_f32 });
        let diag_sqrt_inv = Array2::<f32>::eye(matrix.ncols()) * &diag_sqrt_inv;

        let residual = &matrix - &(Array2::<f32>::eye(diag.len()) * &diag);
        let matrix = diag_sqrt_inv.dot(&residual.dot(&diag_sqrt_inv));
        let (eigen_values, eigen_vectors) = matrix.eigh().unwrap().sort_eig_desc();

        let indices = (0..rank).collect::<Vec<_>>();
        let eigen_values = eigen_values.select(Axis(0), &indices);
        let mut eigen_vectors = eigen_vectors.select(Axis(1), &indices);
        eigen_vectors
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(i, mut r)| {
                let di = diag_sqrt[i];
                r.mapv_inplace(|x| x * di);
            });

        Box::new(LowRankSketch {
            eigen_vectors_diag_scaled: eigen_vectors.t().to_owned(),
            eigen_values,
            diag,
        })
    }
}

#[derive(Serialize, Deserialize)]
pub struct DiagonalSketch {
    diag: Array1<f32>,
}

#[typetag::serde]
impl SymmetricPSDSketch for DiagonalSketch {
    fn dot(&self, u: ArrayView1<f32>) -> f32 {
        self.diag.dot(&u.mapv(|x| x.powi(2)))
    }

    fn size(&self) -> usize {
        self.diag.len()
    }
}

#[derive(Default)]
pub struct DiagonalSketcher {}

impl SymmetricPSDSketcher for DiagonalSketcher {
    fn sketch(&self, matrix: ArrayView2<f32>) -> Box<dyn SymmetricPSDSketch> {
        Box::new(DiagonalSketch {
            diag: matrix.diag().to_owned(),
        })
    }
}
