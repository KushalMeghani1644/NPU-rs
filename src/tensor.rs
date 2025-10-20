use ndarray::{ArrayD, IxDyn};

/// A simple multi-dimensional tensor for our NPU framework.
/// Internally uses `ndarray::ArrayD<f32>` for flexible dimensions.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: ArrayD<f32>,
}

impl Tensor {
    /// Create a new tensor from a Vec and a shape.
    /// Example: Tensor::new(vec![1.0, 2.0, 3.0], &[3])
    pub fn new(data: Vec<f32>, shape: &[usize]) -> Self {
        Self {
            data: ArrayD::from_shape_vec(IxDyn(shape), data)
                .expect("Shape does not match data length"),
        }
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            data: ArrayD::zeros(IxDyn(shape)),
        }
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: &[usize]) -> Self {
        Self {
            data: ArrayD::from_elem(IxDyn(shape), 1.0),
        }
    }

    /// Create a tensor with random values between 0 and 1.
    pub fn random(shape: &[usize]) -> Self {
        use rand::distributions::Uniform;
        use rand::Rng;
        
        let size: usize = shape.iter().product();
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(0.0, 1.0);
        let data: Vec<f32> = (0..size).map(|_| rng.sample(&dist)).collect();
        Self::new(data, shape)
    }

    /// Create a scalar tensor (0-D tensor).
    pub fn from_scalar(value: f32) -> Self {
        Self {
            data: ArrayD::from_elem(IxDyn(&[]), value),
        }
    }

    /// Return the shape of the tensor as a slice.
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Compute the sum of all elements.
    pub fn sum(&self) -> f32 {
        self.data.sum()
    }

    /// Pretty-print tensor contents.
    pub fn print(&self) {
        println!("{:?}", self.data);
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            data: &self.data + &other.data,
        }
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            data: &self.data - &other.data,
        }
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &Self) -> Self {
        Self {
            data: &self.data * &other.data,
        }
    }

    /// Element-wise division.
    pub fn div(&self, other: &Self) -> Self {
        Self {
            data: &self.data / &other.data,
        }
    }

    // === Activation functions ===

    /// ReLU activation function.
    pub fn relu(&self) -> Self {
        Self {
            data: self.data.mapv(|x| if x > 0.0 { x } else { 0.0 }),
        }
    }

    /// Sigmoid activation function.
    pub fn sigmoid(&self) -> Self {
        Self {
            data: self.data.mapv(|x| 1.0 / (1.0 + (-x).exp())),
        }
    }
}
