#[derive(Debug, Default)]
pub struct Graph {
    pub nodes: usize,
    pub edges: usize,
    pub matrix: Vec<Vec<bool>>,
}

#[derive(Debug, Default, Clone)]
pub struct Individual {
    // pub solution: BitVec,
    pub solution: Vec<bool>,
}
impl Individual {
    pub fn new(n: usize) -> Individual {
        Individual {
            solution: vec![false; n],
        }
    }
}
