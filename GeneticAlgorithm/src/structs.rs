use bitvec::vec::BitVec;

#[derive(Debug, Default)]
pub struct Graph {
    pub nodes: usize,
    pub edges: usize,
    pub matrix: Vec<Vec<bool>>
}


#[derive(Debug, Default, Clone)]
pub struct Individual {
    // pub solution: BitVec,
    pub solution: Vec<bool>
}

