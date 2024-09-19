use std::any::type_name;

use criterion::{criterion_group, criterion_main, Criterion};
use stwo_gpu_backend::cuda;
use stwo_gpu_backend::{cuda::BaseFieldVec, CudaBackend};
use stwo_prover::core::fields::m31::BaseField;

use stwo_prover::core::poly::circle::CanonicCoset;
use stwo_prover::core::poly::circle::PolyOps;

pub fn gpu_ntt_base_field(c: &mut Criterion) {
    use stwo_prover::core::backend::Column;

    use criterion::SamplingMode;
    use std::env;

    let full_name = type_name::<BaseField>();

    let group_id = format!("{} ", full_name.rsplit("::").next().unwrap_or(full_name));
    let mut group = c.benchmark_group(&group_id);
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    const MAX_LOG2: u32 = 28; // max length = 2 ^ MAX_LOG2

    let max_log2 = env::var("MAX_LOG2")
        .unwrap_or_else(|_| MAX_LOG2.to_string())
        .parse::<u32>()
        .unwrap_or(MAX_LOG2);

    for test_size_log2 in 8u32..max_log2 + 1 {
        for batch_size_log2 in 0..1 {
            let test_size = 1 << test_size_log2;
            let batch_size = 1 << batch_size_log2;
            let full_size = batch_size * test_size;

            if full_size > 1 << max_log2 {
                continue;
            }

            let cpu_values = (1..(test_size + 1) as u32)
                .map(BaseField::from)
                .collect::<Vec<_>>();
            let gpu_values = BaseFieldVec::from_vec(cpu_values.clone());

            let coset = CanonicCoset::new(test_size_log2 as u32);
            let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, gpu_values);

            let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());

            let len = gpu_evaluations.values.len() as u32;

            let m = gpu_evaluations
                .domain
                .half_coset
                .is_doubling_of(gpu_twiddles.root_coset);
            let f = gpu_evaluations.domain.half_coset.size() as u32;
            let ptr = gpu_evaluations.values.device_ptr;

            let gpu_poly = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);

            for dir in ["interpolate", "evaluate"] {
                let bench_descr = format!(" {} 2 ^ {} = {}", dir, test_size_log2, test_size);
                group.bench_function(&bench_descr, |b| {
                    b.iter(|| {
                        if dir == "evaluate" {
                            CudaBackend::evaluate(&gpu_poly, coset.circle_domain(), &gpu_twiddles);
                        } else {
                            // can't use CudaBackend::evalinterpolate since it's taking ownership of the input and due to .clone() data is copied to device each time
                            assert!(m);
                            unsafe {
                                cuda::bindings::interpolate(
                                    f,
                                    ptr,
                                    gpu_twiddles.itwiddles.device_ptr,
                                    gpu_twiddles.itwiddles.len() as u32,
                                    len,
                                );
                            }
                        }
                    })
                });
            }
        }
    }

    group.finish();
}

criterion_group!(
    name = bit_reverse;
    config = Criterion::default().sample_size(10);
    targets =  gpu_ntt_base_field);
criterion_main!(bit_reverse);
