use stwo_gpu_backend::examples::WideFibonacciComponentCuda;
use stwo_gpu_backend::CudaBackend;
use criterion::{criterion_group, criterion_main, Criterion};

use stwo_gpu_backend::examples::{generate_trace, FibInput, WideFibonacciEvalCuda};

use stwo_prover::constraint_framework::{
    assert_constraints, AssertEvaluator, FrameworkEval, TraceLocationAllocator,
};
use stwo_prover::core::air::Component;
use stwo_prover::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use stwo_prover::core::backend::Column;
use stwo_prover::core::channel::Blake2sChannel;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::pcs::{
    CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig, TreeVec,
};
use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
use stwo_prover::core::poly::BitReversedOrder;
use stwo_prover::core::prover::{prove, verify};
use stwo_prover::core::vcs::blake2_merkle::Blake2sMerkleChannel;
use stwo_prover::core::ColumnVec;

use num_traits::identities::One;
use itertools::Itertools;

const FIB_SEQUENCE_LENGTH: usize = 1024;

fn generate_test_trace(
    log_n_instances: u32,
) -> ColumnVec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>> {
    let inputs = (0..(1 << (log_n_instances - LOG_N_LANES)))
        .map(|i| FibInput {
            a: PackedBaseField::one(),
            b: PackedBaseField::from_array(std::array::from_fn(|j| {
                BaseField::from_u32_unchecked((i * 16 + j) as u32)
            })),
        })
        .collect_vec();
    generate_trace::<FIB_SEQUENCE_LENGTH>(log_n_instances, &inputs)
}

pub fn gpu_fibonacci_prove_verify(c: &mut Criterion) {
    const LOG_N_INSTANCES: u32 = 16;
    let config = PcsConfig::default();
    // Precompute twiddles.
    let twiddles = CudaBackend::precompute_twiddles(
        CanonicCoset::new(LOG_N_INSTANCES + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    // Setup protocol.
    let prover_channel = &mut Blake2sChannel::default();
    let commitment_scheme =
        &mut CommitmentSchemeProver::<CudaBackend, Blake2sMerkleChannel>::new(config, &twiddles);

    // Trace.
    let trace = generate_test_trace(LOG_N_INSTANCES);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(prover_channel);

    // Prove constraints.
    let component = WideFibonacciComponentCuda::new(
        &mut TraceLocationAllocator::default(),
        WideFibonacciEvalCuda::<FIB_SEQUENCE_LENGTH> {
            log_n_rows: LOG_N_INSTANCES,
        },
    );

    let proof = prove::<CudaBackend, Blake2sMerkleChannel>(
        &[&component],
        prover_channel,
        commitment_scheme,
    )
    .unwrap();

    c.bench_function(
        &format!(
            "stwo-gpu Fibonacci fib.prove for {} LOG_N_INSTANCES",
            LOG_N_INSTANCES
        ),
        |b| {
            b.iter(|| {
                let proof = prove::<CudaBackend, Blake2sMerkleChannel>(
                    &[&component],
                    prover_channel,
                    commitment_scheme,
                )
                .unwrap();
            })
        },
    );

    // Verify.
    let verifier_channel = &mut Blake2sChannel::default();
    let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

    // Retrieve the expected column sizes in each commitment interaction, from the AIR.
    let sizes = component.trace_log_degree_bounds();
    commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
    verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();

    // let proof = fib.prove().unwrap();

    // c.bench_function(&format!("stwo-gpu Fibonacci fib.verify for {} LOG_N_INSTANCES", LOG_N_INSTANCES), |b| {
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
