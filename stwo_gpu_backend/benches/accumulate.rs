use criterion::{criterion_group, criterion_main, Criterion};
use stwo_gpu_backend::{cuda::BaseFieldVec, CudaBackend};
use stwo_prover::core::air::accumulation::AccumulationOps;
use stwo_prover::core::fields::m31::M31;
use stwo_prover::core::fields::secure_column::SecureColumnByCoords;

pub fn gpu_accumulate_secure_field(c: &mut Criterion) {
    const BITS: usize = 28;
    let size = 1 << BITS;

    let left_summand: [BaseFieldVec; 4] = [
        BaseFieldVec::from_vec(vec![M31::from(1)].repeat(size)),
        BaseFieldVec::from_vec(vec![M31::from(2)].repeat(size)),
        BaseFieldVec::from_vec(vec![M31::from(3)].repeat(size)),
        BaseFieldVec::from_vec(vec![M31::from(4)].repeat(size)),
    ];
    let right_summand: [BaseFieldVec; 4] = [
        BaseFieldVec::from_vec(vec![M31::from(5)].repeat(size)),
        BaseFieldVec::from_vec(vec![M31::from(6)].repeat(size)),
        BaseFieldVec::from_vec(vec![M31::from(7)].repeat(size)),
        BaseFieldVec::from_vec(vec![M31::from(8)].repeat(size)),
    ];

    let mut left_secure_column = SecureColumnByCoords {
        columns: left_summand,
    };
    let right_secure_column = SecureColumnByCoords {
        columns: right_summand,
    };

    c.bench_function(&format!("gpu accumulate secure_field {} bit", BITS), |b| {
        b.iter(|| {
            CudaBackend::accumulate(&mut left_secure_column, &right_secure_column);
        })
    });
}

criterion_group!(
    name = accumulate;
    config = Criterion::default().sample_size(10);
    targets = gpu_accumulate_secure_field);
criterion_main!(accumulate);
