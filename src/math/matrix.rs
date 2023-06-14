use std::fmt::Debug;
use std::io::{Error, ErrorKind};
use std::iter;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};
use std::process::Output;
use rand::{Rng, thread_rng};
use rand::distributions::uniform::SampleUniform;

trait Numeric:
    Div<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Default
    + Debug
    + PartialOrd
    + SampleUniform
    + Clone
    + Copy
    + Neg<Output = Self>
    + AddAssign
    + SubAssign {}

impl<T> Numeric for T where T:
    Div<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Default
    + Debug
    + PartialOrd
    + SampleUniform
    + Clone
    + Copy
    + Neg<Output = Self>
    + AddAssign
    + SubAssign {}

macro_rules! impl_matrix_ops_for_scalar {
    ($($T:ty),*) => {
        $(
            impl Add<Matrix<$T>> for $T {
                type Output = Matrix<$T>;

                fn add(self, mut matrix: Matrix<$T>) -> Self::Output {
                    matrix.assign(matrix.data.iter().map(|&item| item + self).collect()).unwrap();

                    matrix
                }
            }

            impl Sub<Matrix<$T>> for $T {
                type Output = Matrix<$T>;

                fn sub(self, mut matrix: Matrix<$T>) -> Self::Output {
                    matrix.assign(matrix.data.iter().map(|&item| self - item).collect()).unwrap();

                    matrix
                }
            }

            impl<'a> Sub<&'a mut Matrix<$T>> for $T {
                type Output = Matrix<$T>;

                fn sub(self, mut matrix: &'a mut Matrix<$T>) -> Self::Output {
                    matrix.assign(matrix.data.iter().map(|&item| self - item).collect()).unwrap();

                    matrix.clone()
                }
            }

            impl Mul<Matrix<$T>> for $T {
                type Output = Matrix<$T>;

                fn mul(self, mut matrix: Matrix<$T>) -> Self::Output {
                    matrix.assign(matrix.data.iter().map(|&item| item * self).collect()).unwrap();

                    matrix
                }
            }

            impl Div<Matrix<$T>> for $T {
                type Output = Matrix<$T>;

                fn div(self, mut matrix: Matrix<$T>) -> Self::Output {
                    matrix.assign(matrix.data.iter().map(|&item| self / item).collect()).unwrap();

                    matrix
                }
            }
        )*
    };
}

impl_matrix_ops_for_scalar!(f64, f32, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, usize);

#[derive(Debug, Clone, Default)]
pub struct Shape {
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, Default)]
pub struct Matrix<N> {
    pub shape: Shape,
    pub size: usize,
    data: Vec<N>,
}

impl<N> Matrix<N> where N: Default + Copy + Clone + Debug {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            shape: Shape { rows, cols },
            size: rows * cols,
            data: (0..rows * cols).map(|_| N::default()).collect(),
        }
    }

    pub fn new_empty(rows: usize, cols: usize) -> Self {
        Self {
            shape: Shape { rows, cols },
            size: rows * cols,
            data: vec![],
        }
    }

    pub fn assign(&mut self, data: Vec<N>) -> Result<(), Error> {
        if data.len() != self.shape.rows * self.shape.cols {
            return Err(Error::new(ErrorKind::Other, "Dataset size is different from matrix size"));
        }

        self.data = data;

        Ok(())
    }

    pub fn get_item(&mut self, row: usize, col: usize) -> Option<&mut N> {
        self.data.get_mut(row * self.shape.cols + col)
    }

    pub fn get_row(&mut self, row: usize) -> &mut [N] {
        &mut self.data[row * self.shape.cols .. row * self.shape.cols + self.shape.cols]
    }

    pub fn map(mut self, mut callback: Box<dyn FnMut(N) -> N>) -> Self {
        self.data = self.data.iter().map(|item| callback(item.clone())).collect();
        self
    }

    pub fn all(mut self, mut callback: Box<dyn FnMut(N) -> bool>) -> bool {
        self.data.iter().all(|item| callback(item.clone()))
    }

    pub fn reduce_by_cols(&mut self, mut callback: Box<dyn FnMut(N, N) -> N>) {
        let mut new_data: Vec<N> = vec![];
        for i in (0 .. self.data.len()).step_by(self.shape.cols) {
            let mut item = self.data[i];
            for j in i + 1 .. i + self.shape.cols {
                item = callback(item, self.data[j]);
            }
            new_data.push(item);
        }

        self.data = new_data;
        self.shape.cols = 1;
    }

    pub fn transpose(&mut self) -> Self {
        let mut output = Matrix::new(self.shape.cols, self.shape.rows);
        for i in 0..self.shape.rows {
            for j in 0..self.shape.cols {
                *output.get_item(j, i).unwrap() = self.get_item(i, j).unwrap().clone()
            }
        }
        output
    }

    fn stretch(mut self, axis: i32, size: usize) -> Self {
        let origin = self.data.clone();

        match axis {
            0 => {
                self.shape.rows = size;

                for _ in 0..size {
                    self.data.extend_from_slice(&origin);
                }

                self
            },
            1 => {
                self.shape.cols = size;

                for _ in 0..size {
                    self.data.extend_from_slice(&origin);
                }

                self
            },
            _ => {
                self
            },
        }
    }

    fn handle_stretch(&mut self, matrix: Matrix<N>, action_name: &str) -> Self {
        if Matrix::has_equal_size(&self, &matrix) {
            matrix
        } else {
            if self.shape.cols == matrix.shape.cols && self.shape.rows > matrix.shape.rows && matrix.shape.rows == 1 {
                matrix.stretch(0, self.shape.rows)
            } else if self.shape.rows == matrix.shape.rows && self.shape.cols > matrix.shape.cols && matrix.shape.cols == 1 {
                matrix.stretch(1, self.shape.cols)
            } else {
                let message = format!(
                    "[{:?}] One of matrix dimension - rows or cols should be equals to 1 and another \
                    dimension should be same as origin dimension. \
                    self size: {:?}x{:?} and matrix size: {:?}x{:?})",
                    action_name,
                    self.shape.rows,
                    self.shape.cols,
                    matrix.shape.rows,
                    matrix.shape.cols,
                );
                panic!("{}", message);
            }
        }
    }

    pub fn is_empty(matrix: &Self) -> bool {
        matrix.data.is_empty()
    }

    pub fn has_equal_size(source: &Self, target: &Self) -> bool {
        source.shape.cols == target.shape.cols && source.shape.rows == target.shape.rows
    }
}

impl<N> Matrix<N> where N: Numeric {
    pub fn randomize(mut self, min: N, max: N) -> Self {
        let mut rng = thread_rng();
        self.data = (0..self.shape.rows * self.shape.cols).map(|_| rng.gen_range(min..max)).collect();

        self
    }

    pub fn dot(&mut self, other: &mut Self) -> Self {
        assert_eq!(
            self.shape.cols, other.shape.rows,
            "Cannot multiply matrices A ({:?}) and B ({:?}), \
            please check first matrix cols amount equals to second matrix rows amount",
            self.shape, other.shape
        );

        let mut output = Matrix::new_empty(self.shape.rows, other.shape.cols);

        for i in 0..self.shape.rows {
            for j in 0..output.shape.cols {
                let mut item = N::default();
                for k in 0..other.shape.rows {
                    item += self.get_item(i, k).unwrap().clone() * other.get_item(k, j).unwrap().clone();
                }
                output.data.push(item);
            }
        }

        output
    }

    pub fn sum(&self) -> N {
        self.data.iter().cloned().fold(N::default(), Add::add)
    }

    pub fn x_axis_sum(&mut self) -> Matrix<N> {
        let mut output = Matrix::new(self.shape.rows, 1);
        for i in 0..self.shape.rows {
            *output.get_item(i, 0).unwrap() = self.get_row(i).iter().cloned()
                .fold(N::default(), Add::add);
        }

        output
    }

    pub fn y_axis_sum(&mut self) -> Matrix<N> {
        let mut output = Matrix::new(1, self.shape.cols);
        for j in 0..self.shape.cols {
            *output.get_item(0, j).unwrap() = (0..self.shape.rows)
                .map(|i| *self.get_item(i, j).unwrap())
                .fold(N::default(), Add::add);
        }

        output
    }
}

impl<N: PartialEq> PartialEq for Matrix<N> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
            && self.shape.rows == other.shape.rows
            && self.shape.cols == other.shape.cols
    }
}

impl<N> Neg for Matrix<N> where N: Numeric {
    type Output = Matrix<N>;

    fn neg(self) -> Self::Output {
        self.map(Box::new(|x| -x))
    }
}

impl<N> SubAssign<N> for Matrix<N> where N: Numeric {
    fn sub_assign(&mut self, scalar: N) {
        self.assign(self.data.iter().map(|&item| item - scalar).collect()).unwrap();
    }
}

impl<N> SubAssign<Matrix<N>> for Matrix<N> where N: Numeric {
    fn sub_assign(&mut self, mut matrix: Self) {
        let mut matrix = Matrix::handle_stretch(self, matrix, "SubAssign matrices");
        let mut items = vec![];
        for i in 0..self.shape.rows {
            for j in 0..self.shape.cols {
                items.push(*self.get_item(i, j).unwrap() - *matrix.get_item(i, j).unwrap());
            }
        }
        self.assign(items).unwrap();
    }
}

impl<N> Add<N> for Matrix<N> where N: Numeric {
    type Output = Matrix<N>;

    fn add(self, scalar: N) -> Self::Output {
        let mut output = Matrix::new(self.shape.rows, self.shape.cols);
        output.assign(self.data.iter().map(|&item| item + scalar).collect()).unwrap();

        output
    }
}

impl<N> Add<Matrix<N>> for Matrix<N> where N: Numeric {
    type Output = Matrix<N>;

    fn add(mut self, mut matrix: Matrix<N>) -> Self::Output {
        let mut matrix = Matrix::handle_stretch(&mut self, matrix, "Add");
        let mut output = Matrix::new(self.shape.rows, self.shape.cols);
        let mut items = vec![];
        for i in 0..self.shape.rows {
            for j in 0..self.shape.cols {
                items.push(*self.get_item(i, j).unwrap() + *matrix.get_item(i, j).unwrap());
            }
        }
        output.assign(items).unwrap();

        output
    }
}

impl<N> Sub<N> for Matrix<N> where N: Numeric {
    type Output = Matrix<N>;

    fn sub(self, scalar: N) -> Self::Output {
        let mut output = Matrix::new(self.shape.rows, self.shape.cols);
        output.assign(self.data.iter().map(|&item| item - scalar).collect()).unwrap();

        output
    }
}

impl<N> Sub<Matrix<N>> for Matrix<N> where N: Numeric {
    type Output = Matrix<N>;

    fn sub(mut self, mut matrix: Matrix<N>) -> Self::Output {
        let mut matrix = Matrix::handle_stretch(&mut self, matrix, "Sub matrices");
        let mut output = Matrix::new(self.shape.rows, self.shape.cols);
        let mut items = vec![];
        for i in 0..self.shape.rows {
            for j in 0..self.shape.cols {
                items.push(*self.get_item(i, j).unwrap() - *matrix.get_item(i, j).unwrap());
            }
        }
        output.assign(items).unwrap();

        output
    }
}

impl<N> Mul<N> for Matrix<N> where N: Numeric {
    type Output = Matrix<N>;

    fn mul(self, scalar: N) -> Self::Output {
        let mut output = Matrix::new(self.shape.rows, self.shape.cols);
        output.assign(self.data.iter().map(|&item| item * scalar).collect()).unwrap();

        output
    }
}

impl<N> Mul<Matrix<N>> for Matrix<N> where N: Numeric {
    type Output = Matrix<N>;

    fn mul(mut self, mut matrix: Matrix<N>) -> Self::Output {
        let mut matrix = Matrix::handle_stretch(&mut self, matrix, "Mul matrices");
        let mut output = Matrix::new(self.shape.rows, self.shape.cols);
        let mut items = vec![];
        for i in 0..self.shape.rows {
            for j in 0..self.shape.cols {
                items.push(*self.get_item(i, j).unwrap() * *matrix.get_item(i, j).unwrap());
            }
        }
        output.assign(items).unwrap();

        output
    }
}

impl<N> Div<N> for Matrix<N> where N: Numeric {
    type Output = Matrix<N>;

    fn div(self, scalar: N) -> Self::Output {
        let mut output = Matrix::new(self.shape.rows, self.shape.cols);
        output.assign(self.data.iter().map(|&item| item / scalar).collect()).unwrap();

        output
    }
}

impl<N> Div<Matrix<N>> for Matrix<N> where N: Numeric {
    type Output = Matrix<N>;

    fn div(mut self, mut matrix: Matrix<N>) -> Self::Output {
        let mut matrix = Matrix::handle_stretch(&mut self, matrix, "Div matrices");
        let mut output = Matrix::new(self.shape.rows, self.shape.cols);
        let mut items = vec![];
        for i in 0..self.shape.rows {
            for j in 0..self.shape.cols {
                items.push(*self.get_item(i, j).unwrap() / *matrix.get_item(i, j).unwrap());
            }
        }
        output.assign(items).unwrap();

        output
    }
}

impl<N> Div<&mut Matrix<N>> for &mut Matrix<N> where N: Numeric {
    type Output = Matrix<N>;

    fn div(mut self, matrix: &mut Matrix<N>) -> Self::Output {
        let mut matrix = Matrix::handle_stretch(&mut self, matrix.clone(), "Div &matrices");
        let mut output = Matrix::new(self.shape.rows, self.shape.cols);
        let mut items = vec![];
        for i in 0..self.shape.rows {
            for j in 0..self.shape.cols {
                items.push(*self.get_item(i, j).unwrap() / *matrix.get_item(i, j).unwrap());
            }
        }
        output.assign(items).unwrap();

        output
    }
}

#[cfg(test)]
mod tests {
    use crate::math::matrix::{Matrix};

    const MATRIX_SAMPLE_2X2: [f64; 4] = [1., 2., 3., 4.];
    const MATRIX_SAMPLE_3X2: [f64; 6] = [1., 2., 3., 4., 5., 6.];
    const MATRIX_SAMPLE_2X3: [f64; 6] = [1., 2., 3., 4., 5., 6.];
    const MATRIX_SAMPLE_3X1: [f64; 3] = [1., 2., 3.];
    const MATRIX_SAMPLE_3X3: [f64; 9] = [1., 2., 3., 4., 5., 6., 7., 8., 9.];

    const ROWS_2: usize = 2;
    const ROWS_3: usize = 3;
    const COLS_1: usize = 1;
    const COLS_2: usize = 2;
    const COLS_3: usize = 3;

    #[test]
    fn test_matrix_creation() {
        let mut matrix: Matrix<f64> = Matrix::new(ROWS_2, COLS_2);
        matrix.assign(MATRIX_SAMPLE_2X2.to_vec()).unwrap();

        assert_eq!(matrix.data, MATRIX_SAMPLE_2X2.to_vec());
    }

    #[test]
    fn test_matrices_equality() {
        let mut matrix_a: Matrix<f64> = Matrix::new(ROWS_2, COLS_2);
        let mut matrix_b: Matrix<f64> = Matrix::new(ROWS_2, COLS_2);
        let mut matrix_c: Matrix<f64> = Matrix::new(ROWS_2, COLS_3);

        matrix_a.assign(MATRIX_SAMPLE_2X2.to_vec()).unwrap();
        matrix_b.assign(MATRIX_SAMPLE_2X2.to_vec()).unwrap();
        matrix_c.assign(MATRIX_SAMPLE_2X3.to_vec()).unwrap();

        assert_eq!(matrix_a, matrix_b);
        assert_ne!(matrix_a, matrix_c);
    }

    #[test]
    fn test_two_matrix_with_same_size_addition() {
        let mut matrix_a: Matrix<f64> = Matrix::new(ROWS_2, COLS_2);
        let mut matrix_b: Matrix<f64> = Matrix::new(ROWS_2, COLS_2);

        matrix_a.assign(MATRIX_SAMPLE_2X2.to_vec()).unwrap();
        matrix_b.assign(MATRIX_SAMPLE_2X2.to_vec()).unwrap();

        let matrix_c: Matrix<f64> = matrix_a + matrix_b;

        assert_eq!(matrix_c.data, vec![2.0, 4.0, 6.0, 8.0]);
        assert_eq!(matrix_c.shape.rows, ROWS_2);
        assert_eq!(matrix_c.shape.cols, COLS_2);
    }

    #[test]
    #[should_panic]
    fn test_two_matrix_with_wrong_size_addition() {
        let mut matrix_a: Matrix<f64> = Matrix::new(ROWS_2, COLS_2);
        let mut matrix_b: Matrix<f64> = Matrix::new(ROWS_3, COLS_2);

        matrix_a.assign(MATRIX_SAMPLE_2X2.to_vec()).unwrap();
        matrix_b.assign(MATRIX_SAMPLE_3X2.to_vec()).unwrap();

        let matrix_c: Matrix<f64> = matrix_a + matrix_b;
    }

    #[test]
    fn test_two_matrix_with_same_size_multiplication() {
        let mut matrix_a: Matrix<f64> = Matrix::new(ROWS_2, COLS_2);
        let mut matrix_b: Matrix<f64> = Matrix::new(ROWS_2, COLS_2);

        matrix_a.assign(MATRIX_SAMPLE_2X2.to_vec()).unwrap();
        matrix_b.assign(MATRIX_SAMPLE_2X2.to_vec()).unwrap();

        let matrix_c: Matrix<f64> = matrix_a.dot(&mut matrix_b);
        assert_eq!(matrix_c.data, vec![7.0, 10.0, 15.0, 22.0]);
        assert_eq!(matrix_c.shape.rows, ROWS_2);
        assert_eq!(matrix_c.shape.cols, COLS_2);
    }

    #[test]
    fn test_two_matrix_with_different_size_multiplication() {
        let mut matrix_a: Matrix<f64> = Matrix::new(ROWS_3, COLS_3);
        let mut matrix_b: Matrix<f64> = Matrix::new(ROWS_3, COLS_1);

        matrix_a.assign(MATRIX_SAMPLE_3X3.to_vec()).unwrap();
        matrix_b.assign(MATRIX_SAMPLE_3X1.to_vec()).unwrap();

        let matrix_c: Matrix<f64> = matrix_a.dot(&mut matrix_b);
        assert_eq!(matrix_c.data, vec![14.0, 32.0, 50.0]);
        assert_eq!(matrix_c.shape.rows, ROWS_3);
        assert_eq!(matrix_c.shape.cols, COLS_1);
    }

    #[test]
    #[should_panic]
    fn test_two_matrix_with_wrong_size_multiplication() {
        let mut matrix_a: Matrix<f64> = Matrix::new(ROWS_2, COLS_3);
        let mut matrix_b: Matrix<f64> = Matrix::new(ROWS_2, COLS_2);

        matrix_a.assign(MATRIX_SAMPLE_2X3.to_vec()).unwrap();
        matrix_b.assign(MATRIX_SAMPLE_2X2.to_vec()).unwrap();

        let matrix_c = matrix_a.dot(&mut matrix_b);
    }

    #[test]
    #[should_panic]
    fn test_wrong_data_size_matrix_assign() {
        let mut matrix_a: Matrix<f64> = Matrix::new(ROWS_2, COLS_2);
        matrix_a.assign(MATRIX_SAMPLE_2X3.to_vec()).unwrap();
    }

    #[test]
    fn test_get_matrix_single_item() {
        let mut matrix_a: Matrix<f64> = Matrix::new(ROWS_2, COLS_2);
        matrix_a.assign(MATRIX_SAMPLE_2X2.to_vec()).unwrap();

        let item = *matrix_a.get_item(0, 1).unwrap();
        assert_eq!(item, MATRIX_SAMPLE_2X2[1]);
    }

    #[test]
    fn test_get_row() {
        let mut matrix_a: Matrix<f64> = Matrix::new(ROWS_2, COLS_2);
        matrix_a.assign(MATRIX_SAMPLE_2X2.to_vec()).unwrap();

        assert_eq!(matrix_a.get_row(0), vec![1., 2.]);
        assert_eq!(matrix_a.get_row(1), vec![3., 4.]);

        let mut matrix_a: Matrix<f64> = Matrix::new(ROWS_3, COLS_1);
        matrix_a.assign(MATRIX_SAMPLE_3X1.to_vec()).unwrap();

        assert_eq!(matrix_a.get_row(0), vec![1.]);
        assert_eq!(matrix_a.get_row(1), vec![2.]);
        assert_eq!(matrix_a.get_row(2), vec![3.]);

        let mut matrix_a: Matrix<f64> = Matrix::new(ROWS_2, COLS_3);
        matrix_a.assign(MATRIX_SAMPLE_2X3.to_vec()).unwrap();

        assert_eq!(matrix_a.get_row(0), vec![1., 2., 3.]);
        assert_eq!(matrix_a.get_row(1), vec![4., 5., 6.]);

        let mut matrix_a: Matrix<f64> = Matrix::new(ROWS_3, COLS_3);
        matrix_a.assign(MATRIX_SAMPLE_3X3.to_vec()).unwrap();

        assert_eq!(matrix_a.get_row(0), vec![1., 2., 3.]);
        assert_eq!(matrix_a.get_row(1), vec![4., 5., 6.]);
        assert_eq!(matrix_a.get_row(2), vec![7., 8., 9.]);
    }

    #[test]
    fn test_update_row() {
        let mut matrix_a: Matrix<f64> = Matrix::new(ROWS_3, COLS_1);
        matrix_a.assign(MATRIX_SAMPLE_3X1.to_vec()).unwrap();

        assert_eq!(matrix_a.get_row(0), vec![1.]);
        assert_eq!(matrix_a.get_row(1), vec![2.]);
        assert_eq!(matrix_a.get_row(2), vec![3.]);

        matrix_a.get_row(0)[0] = 5.;

        assert_eq!(matrix_a.get_row(0), vec![5.]);
    }
}