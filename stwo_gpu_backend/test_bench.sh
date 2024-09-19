# LOG_N_INSTANCES=18 RUST_LOG_SPAN_EVENTS=close RUST_BACKTRACE=1 RUST_LOG=info cargo test evaluate  --release --features icicle_poc -- --nocapture |& tee test_release_icicle.log
# profile
LOG_N_INSTANCES=18 RUST_LOG_SPAN_EVENTS=close RUST_BACKTRACE=1 RUST_LOG=info nsys profile --trace=cuda,cublas,osrt --output=nsys_profile_test_release_icicle cargo test evaluate --release --features icicle_poc -- --nocapture |& tee test_release_icicle.log
