use criterion::{criterion_group, criterion_main, Criterion};
use stwo_prover::core::fields::m31::BaseField;

use stwo_gpu_backend::examples::Fibonacci;


pub fn gpu_fibonacci_prove_verify(c: &mut Criterion) {
    const FIB_LOG_SIZE: u32 = 23;
    let fib = Fibonacci::new(FIB_LOG_SIZE, BaseField::from(729665688));

    c.bench_function(&format!("stwo-gpu Fibonacci fib.prove for {} FIB_LOG_SIZE", FIB_LOG_SIZE), |b| {
        b.iter(|| {
            fib.prove().unwrap();
        })
    });
    
    // let proof = fib.prove().unwrap();

    // c.bench_function(&format!("stwo-gpu Fibonacci fib.verify for {} FIB_LOG_SIZE", FIB_LOG_SIZE), |b| {
    //     b.iter(|| {
    //         fib.verify(proof).unwrap();
    //     })
    // });
}

criterion_group!(
    name = fibonacci;
    config = Criterion::default().sample_size(10);
    targets = gpu_fibonacci_prove_verify);
criterion_main!(fibonacci);
