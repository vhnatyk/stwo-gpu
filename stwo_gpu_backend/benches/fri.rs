use criterion::{black_box, criterion_group, criterion_main, Criterion};

use stwo_gpu_backend::CudaBackend;

use stwo_prover::core::backend::{Backend, ColumnOps, CpuBackend};
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::fields::secure_column::SecureColumnByCoords;
use stwo_prover::core::fri::FriOps;
use stwo_prover::core::poly::circle::{CanonicCoset, PolyOps, SecureEvaluation};
use stwo_prover::core::poly::line::{LineDomain, LineEvaluation};
use stwo_prover::core::poly::twiddles::TwiddleTree;
use stwo_prover::core::poly::BitReversedOrder;

use stwo_gpu_backend::cuda::BaseFieldVec;

fn folding_benchmark(c: &mut Criterion) {
    for log_size in 1..=20 {
        let values: Vec<SecureField> = (0..(1 << log_size))
            .map(|i| SecureField::from_u32_unchecked(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3))
            .collect();
        let alpha = SecureField::from_u32_unchecked(1, 3, 5, 7);

        let mut vec: [Vec<BaseField>; 4] = [vec![], vec![], vec![], vec![]];
        values.iter().for_each(|a| {
            vec[0].push(BaseField::from_u32_unchecked(a.0 .0 .0));
            vec[1].push(BaseField::from_u32_unchecked(a.0 .1 .0));
            vec[2].push(BaseField::from_u32_unchecked(a.1 .0 .0));
            vec[3].push(BaseField::from_u32_unchecked(a.1 .1 .0));
        });

        let vecs = [
            BaseFieldVec::from_vec(vec[0].clone()),
            BaseFieldVec::from_vec(vec[1].clone()),
            BaseFieldVec::from_vec(vec[2].clone()),
            BaseFieldVec::from_vec(vec[3].clone()),
        ];

        // fold line
        let line_domain = LineDomain::new(CanonicCoset::new(log_size + 1).half_coset());

        fn bench_fold_line<B: Backend + FriOps>(
            log_size: u32,
            line_domain: LineDomain,
            evals: LineEvaluation<B>,
            c: &mut Criterion,
            alpha: SecureField,
            backend_descr: &str,
        ) {
            let twiddles = B::precompute_twiddles(line_domain.coset());

            c.bench_function(
                &format!("{} fold_line log2 = {}", backend_descr, log_size),
                |b| {
                    b.iter(|| {
                        black_box(B::fold_line(black_box(&evals), black_box(alpha), &twiddles));
                    })
                },
            );
        }

        let cpu_evals = LineEvaluation::new(
            line_domain,
            SecureColumnByCoords {
                columns: std::array::from_fn(|i| {
                    vec![BaseField::from_u32_unchecked(i as u32); 1 << log_size]
                }),
            },
        );

        bench_fold_line::<CpuBackend>(log_size, line_domain, cpu_evals, c, alpha, "cpu");

        let gpu_evals = LineEvaluation::new(
            line_domain,
            SecureColumnByCoords {
                columns: vecs.clone(),
            },
        );

        bench_fold_line::<CudaBackend>(log_size, line_domain, gpu_evals, c, alpha, "stwo_gpu");

        // fold circle

        let circle_domain = CanonicCoset::new(log_size).circle_domain();

        let line_domain = LineDomain::new(circle_domain.half_coset);

        fn bench_fold_circle<B: Backend + FriOps + ColumnOps<BaseField>>(
            log_size: u32,
            line_domain: LineDomain,
            c: &mut Criterion,
            src: &SecureEvaluation<B, BitReversedOrder>,
            alpha: SecureField,
            backend_descr: &str,
        ) {
            let twiddles = B::precompute_twiddles(line_domain.coset());

            let mut dst =
                LineEvaluation::new(line_domain, SecureColumnByCoords::zeros(1 << log_size - 1));

            c.bench_function(
                &format!(
                    "{} fold_circle_into_line log2 = {}",
                    backend_descr, log_size
                ),
                |b| {
                    b.iter(|| {
                        black_box(B::fold_circle_into_line(
                            black_box(&mut dst),
                            black_box(&src),
                            black_box(alpha),
                            black_box(&twiddles),
                        ));
                    })
                },
            );
        }

        let cpu_src = SecureEvaluation::new(
            circle_domain,
            SecureColumnByCoords {
                columns: vec.clone(),
            },
        );

        bench_fold_circle::<CpuBackend>(log_size, line_domain, c, &cpu_src, alpha, "cpu");

        let gpu_src = SecureEvaluation::new(
            circle_domain,
            SecureColumnByCoords {
                columns: vecs.clone(),
            },
        );

        bench_fold_circle::<CudaBackend>(log_size, line_domain, c, &gpu_src, alpha, "stwo_gpu");
    }
}

criterion_group!(benches, folding_benchmark);
criterion_main!(benches);
