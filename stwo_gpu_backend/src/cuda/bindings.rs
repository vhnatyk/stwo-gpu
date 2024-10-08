use stwo_prover::core::vcs::blake2_hash::Blake2sHash;
use stwo_prover::core::{
    circle::CirclePoint,
    fields::{m31::BaseField, qm31::SecureField},
};
use tracing::{info, span, trace, Level};
use std::ffi::c_void;

#[repr(C)]
#[derive(Debug)]
pub struct CudaSecureField {
    a: BaseField,
    b: BaseField,
    c: BaseField,
    d: BaseField,
}

impl CudaSecureField {
    pub unsafe fn zero() -> Self {
        Self {
            a: BaseField::from(0),
            b: BaseField::from(0),
            c: BaseField::from(0),
            d: BaseField::from(0),
        }
    }
}

impl From<SecureField> for CudaSecureField {
    fn from(value: SecureField) -> Self {
        Self {
            a: value.0 .0,
            b: value.0 .1,
            c: value.1 .0,
            d: value.1 .1,
        }
    }
}

impl From<CudaSecureField> for SecureField{
    fn from(value: CudaSecureField) -> Self {
        SecureField::from_m31(value.a, value.b, value.c, value.d)
    }
}

#[repr(C)]
pub(crate) struct CirclePointBaseField {
    x: BaseField,
    y: BaseField,
}

impl From<CirclePoint<BaseField>> for CirclePointBaseField {
    fn from(value: CirclePoint<BaseField>) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

#[repr(C)]
pub(crate) struct CirclePointSecureField {
    x: CudaSecureField,
    y: CudaSecureField,
}

impl From<CirclePoint<SecureField>> for CirclePointSecureField {
    fn from(value: CirclePoint<SecureField>) -> Self {
        Self {
            x: CudaSecureField::from(value.x),
            y: CudaSecureField::from(value.y),
        }
    }
}

#[link(name = "gpubackend")]
extern "C" {
    #[link_name = "copy_uint32_t_vec_from_device_to_host"]
    fn copy_uint32_t_vec_from_device_to_host_cuda(
        device_ptr: *const u32,
        host_ptr: *const u32,
        size: u32,
    );

    #[link_name = "copy_uint32_t_vec_from_host_to_device"]
    fn copy_uint32_t_vec_from_host_to_device_cuda(host_ptr: *const u32, size: u32) -> *const u32;

    #[link_name = "copy_uint32_t_vec_from_device_to_device"]
    fn copy_uint32_t_vec_from_device_to_device_cuda(
        from: *const u32,
        dst: *const u32,
        size: u32,
    ) -> *const u32;

    #[link_name = "cuda_malloc_uint32_t"]
    fn cuda_malloc_uint32_t_cuda(size: u32) -> *const u32;

    #[link_name = "cuda_malloc_blake_2s_hash"]
    fn cuda_malloc_blake_2s_hash_cuda(size: usize) -> *const Blake2sHash;

    #[link_name = "cuda_alloc_zeroes_uint32_t"]
    fn cuda_alloc_zeroes_uint32_t_cuda(size: u32) -> *const u32;

    #[link_name = "cuda_alloc_zeroes_blake_2s_hash"]
    fn cuda_alloc_zeroes_blake_2s_hash_cuda(size: usize) -> *const Blake2sHash;

    #[link_name = "free_uint32_t_vec"]
    fn free_uint32_t_vec_cuda(device_ptr: *const u32);

    pub fn cuda_free_memory(device_ptr: *const c_void);

    #[cfg(not(feature = "icicle_poc"))]
    #[link_name = "bit_reverse_base_field"]
    fn bit_reverse_base_field_cuda(array: *const u32, size: usize);

    #[cfg(not(feature = "icicle_poc"))]
    #[link_name = "bit_reverse_secure_field"]
    fn bit_reverse_secure_field_cuda(array: *const u32, size: usize);

    #[link_name = "batch_inverse_base_field"]
    fn batch_inverse_base_field_cuda(from: *const u32, dst: *const u32, size: usize);

    #[link_name = "batch_inverse_secure_field"]
    fn batch_inverse_secure_field_cuda(from: *const u32, dst: *const u32, size: usize);

    #[link_name = "sort_values_and_permute_with_bit_reverse_order"]
    fn sort_values_and_permute_with_bit_reverse_order_cuda(
        from: *const u32,
        size: usize,
    ) -> *const u32;

    #[link_name = "precompute_twiddles"]
    fn precompute_twiddles_cuda(
        initial: CirclePointBaseField,
        step: CirclePointBaseField,
        total_size: usize,
    ) -> *const u32;

    #[link_name = "interpolate"]
    fn interpolate_cuda(
        eval_domain_size: u32,
        values: *const u32,
        inverse_twiddles_tree: *const u32,
        inverse_twiddle_tree_size: u32,
        values_size: u32,
    );

    #[link_name = "evaluate"]
    fn evaluate_cuda(
        eval_domain_size: u32,
        values: *const u32,
        twiddles_tree: *const u32,
        twiddle_tree_size: u32,
        values_size: u32,
    );

    pub fn interpolate_columns(
        eval_domain_size: u32,
        values: *const *const u32,
        inverse_twiddles_tree: *const u32,
        inverse_twiddle_tree_size: u32,
        values_size: u32,
        number_of_rows: u32,
    );

    pub fn evaluate_columns(
        eval_domain_sizes: *const u32,
        values: *const *const u32,
        twiddles_tree: *const u32,
        twiddle_tree_size: u32,
        number_of_columns: u32,
        column_sizes: *const u32,
    );

    #[link_name = "eval_at_point"]
    fn eval_at_point_cuda(
        coeffs: *const u32,
        coeffs_size: u32,
        point_x: CudaSecureField,
        point_y: CudaSecureField,
    ) -> CudaSecureField;

    #[link_name = "fold_line"]
    fn fold_line_cuda(
        gpu_domain: *const u32,
        twiddle_offset: usize,
        n: usize,
        eval_values: *const *const u32,
        alpha: CudaSecureField,
        folded_values: *const *const u32,
    );

    #[link_name = "fold_circle_into_line"]
    fn fold_circle_into_line_cuda(
        gpu_domain: *const u32,
        twiddle_offset: usize,
        n: usize,
        eval_values: *const *const u32,
        alpha: CudaSecureField,
        folded_values: *const *const u32,
    );

    #[link_name = "decompose"]
    fn decompose_cuda(
        columns: *const *const u32,
        column_size: u32,
        lambda: &CudaSecureField,
        g_values: *const *const u32,
    );

    #[link_name = "accumulate"]
    fn accumulate_cuda(
        size: u32,
        left_columns: *const *const u32,
        right_columns: *const *const u32,
    );

    #[link_name = "commit_on_first_layer"]
    fn commit_on_first_layer_cuda(
        size: usize,
        amount_of_columns: usize,
        columns: *const *const u32,
        result: *mut Blake2sHash,
    );

    #[link_name = "commit_on_layer_with_previous"]
    fn commit_on_layer_with_previous_cuda(
        size: usize,
        amount_of_columns: usize,
        columns: *const *const u32,
        previous_layer: *const Blake2sHash,
        result: *mut Blake2sHash,
    );

    #[link_name = "copy_blake_2s_hash_vec_from_host_to_device"]
    fn copy_blake_2s_hash_vec_from_host_to_device_cuda(
        from: *const Blake2sHash,
        size: usize,
    ) -> *mut Blake2sHash;

    #[link_name = "copy_blake_2s_hash_vec_from_device_to_host"]
    fn copy_blake_2s_hash_vec_from_device_to_host_cuda(
        from: *const Blake2sHash,
        to: *const Blake2sHash,
        size: usize,
    );

    #[link_name = "copy_blake_2s_hash_vec_from_device_to_device"]
    fn copy_blake_2s_hash_vec_from_device_to_device_cuda(
        from: *const Blake2sHash,
        dst: *const Blake2sHash,
        size: usize,
    );

    #[link_name = "free_blake_2s_hash_vec"]
    fn free_blake_2s_hash_vec_cuda(device_pointer: *const Blake2sHash);

    #[link_name = "copy_device_pointer_vec_from_host_to_device"]
    fn copy_device_pointer_vec_from_host_to_device_cuda(
        from: *const *const u32,
        size: usize,
    ) -> *const *const u32;

    #[link_name = "free_device_pointer_vec"]
    fn free_device_pointer_vec_cuda(device_pointer: *const *const u32);

    #[link_name = "accumulate_quotients"]
    fn accumulate_quotients_cuda(
        half_coset_initial_index: u32,
        half_coset_step_size: u32,
        domain_size: u32,
        columns: *const *const u32,
        number_of_columns: usize,
        random_coeff: CudaSecureField,
        sample_points: *const u32,
        sample_columns_indexes: *const u32,
        sample_columns_indexes_size: u32,
        sample_column_values: *const CudaSecureField,
        sample_column_and_values_sizes: *const u32,
        sample_size: u32,
        result_column_0: *const u32,
        result_column_1: *const u32,
        result_column_2: *const u32,
        result_column_3: *const u32,
        flattened_line_coeffs_size: u32,
    );

    #[link_name = "fibonacci_component_evaluate_constraint_quotients_on_domain"]
    fn fibonacci_component_evaluate_constraint_quotients_on_domain_cuda(
        evals: *const u32,
        evals_size: u32,
        output_column_0: *const u32,
        output_column_1: *const u32,
        output_column_2: *const u32,
        output_column_3: *const u32,
        claim_value: BaseField,
        initial_point: CirclePointBaseField,
        step_point: CirclePointBaseField,
        random_coeff_0: CudaSecureField,
        random_coeff_1: CudaSecureField,
    );

    pub fn gen_eq_evals(
        v: CudaSecureField,
        y: *const CudaSecureField,
        y_size: u32,
        evals: *const CudaSecureField,
        evals_size: u32,
    );

    pub fn fix_first_variable_base_field(
        evals: *const u32,
        evals_size: usize,
        assignment: CudaSecureField,
        output_evals: *const u32,
    );

    pub fn fix_first_variable_secure_field(
        evals: *const u32,
        evals_size: usize,
        assignment: CudaSecureField,
        output_evals: *const u32,
    );
}

// Wrappers with logging

pub unsafe fn copy_uint32_t_vec_from_device_to_host(
    device_ptr: *const u32,
    host_ptr: *const u32,
    size: u32,
) {
    let _ = span!(
        Level::TRACE,
        " <<<TO HOST==: copy_uint32_t_vec_from_device_to_host",
        message = format!("called with size: {:?}", size)
    )
    .entered();
    unsafe {
        copy_uint32_t_vec_from_device_to_host_cuda(device_ptr, host_ptr, size);
    }
}

pub unsafe fn copy_uint32_t_vec_from_host_to_device(host_ptr: *const u32, size: u32) -> *const u32 {
    let _ = span!(
        Level::TRACE,
        " ==TO DEVICE>>: copy_uint32_t_vec_from_host_to_device",
        message = format!("called with size: {:?}", size)
    )
    .entered();
    unsafe { copy_uint32_t_vec_from_host_to_device_cuda(host_ptr, size) }
}

pub unsafe fn copy_uint32_t_vec_from_device_to_device(
    from: *const u32,
    dst: *const u32,
    size: u32,
) -> *const u32 {
    let _ = span!(
        Level::TRACE,
        "==DtoD==: copy_uint32_t_vec_from_device_to_device",
        message = format!("called with size: {:?}", size)
    )
    .entered();
    unsafe { copy_uint32_t_vec_from_device_to_device_cuda(from, dst, size) }
}

pub unsafe fn cuda_malloc_uint32_t(size: u32) -> *const u32 {
    trace!(
        target = "cuda_malloc_uint32_t",
        message = format!("called with size: {:?}", size)
    );
    unsafe { cuda_malloc_uint32_t_cuda(size) }
}

pub unsafe fn cuda_malloc_blake_2s_hash(size: usize) -> *const Blake2sHash {
    info!(
        target = "cuda_malloc_blake_2s_hash",
        message = format!("called with size: {:?}", size)
    );
    unsafe { cuda_malloc_blake_2s_hash_cuda(size) }
}

pub unsafe fn cuda_alloc_zeroes_uint32_t(size: u32) -> *const u32 {
    info!(
        target = "cuda_alloc_zeroes_uint32_t",
        message = format!("called with size: {:?}", size)
    );
    unsafe { cuda_alloc_zeroes_uint32_t_cuda(size) }
}

pub unsafe fn cuda_alloc_zeroes_blake_2s_hash(size: usize) -> *const Blake2sHash {
    info!(
        target = "cuda_alloc_zeroes_blake_2s_hash",
        message = format!("called with size: {:?}", size)
    );
    unsafe { cuda_alloc_zeroes_blake_2s_hash_cuda(size) }
}

pub unsafe fn free_uint32_t_vec(device_ptr: *const u32) {
    info!("free_uint32_t_vec called");
    unsafe {
        free_uint32_t_vec_cuda(device_ptr);
    }
}

pub unsafe fn bit_reverse_base_field(array: *const u32, size: usize) {
    let _ = span!(
        Level::INFO,
        "bit_reverse_base_field",
        message = format!("called with size: {:?}", size)
    )
    .entered();

    #[cfg(not(feature = "icicle_poc"))]
    unsafe {
        bit_reverse_base_field_cuda(array, size)
    }

    #[cfg(feature = "icicle_poc")]
    unsafe {
        use icicle_core::vec_ops::BitReverseConfig;

        let cfg = BitReverseConfig::default();

        use icicle_core::vec_ops::bit_reverse_inplace;
        use icicle_m31::field::ScalarField;
        use icicle_cuda_runtime::memory::DeviceSlice;
        use std::slice;
        let ptr = array as *mut ScalarField;
        let rr = slice::from_raw_parts_mut(ptr, size);
        bit_reverse_inplace::<ScalarField>(DeviceSlice::from_mut_slice(rr), &cfg).unwrap();
    }
}

pub unsafe fn bit_reverse_secure_field(array: *const u32, size: usize) {
    let _ = span!(
        Level::INFO,
        "bit_reverse_secure_field",
        message = format!("called with size: {:?}", size)
    )
    .entered();
    #[cfg(not(feature = "icicle_poc"))]
    unsafe {
        bit_reverse_secure_field_cuda(array, size)
    }

    #[cfg(feature = "icicle_poc")]
    unsafe {
        use icicle_core::vec_ops::BitReverseConfig;

        let cfg = BitReverseConfig::default();

        use icicle_core::vec_ops::bit_reverse_inplace;
        use icicle_m31::field::ExtensionField;
        use icicle_cuda_runtime::memory::DeviceSlice;
        use std::slice;
        let ptr = array as *mut ExtensionField;
        let rr = slice::from_raw_parts_mut(ptr, size);
        bit_reverse_inplace::<ExtensionField>(DeviceSlice::from_mut_slice(rr), &cfg).unwrap();
    }
}

pub unsafe fn batch_inverse_base_field(from: *const u32, dst: *const u32, size: usize) {
    let _ = span!(
        Level::INFO,
        "batch_inverse_base_field",
        message = format!("called with size: {:?}", size)
    )
    .entered();
    unsafe { batch_inverse_base_field_cuda(from, dst, size) }
}

pub unsafe fn batch_inverse_secure_field(from: *const u32, dst: *const u32, size: usize) {
    let _ = span!(
        Level::INFO,
        "batch_inverse_secure_field",
        message = format!("called with size: {:?}", size)
    )
    .entered();
    unsafe { batch_inverse_secure_field_cuda(from, dst, size) }
}

pub unsafe fn sort_values_and_permute_with_bit_reverse_order(
    from: *const u32,
    size: usize,
) -> *const u32 {
    let _ = span!(
        Level::INFO,
        "sort_values_and_permute_with_bit_reverse_order",
        message = format!("called with size: {:?}", size)
    )
    .entered();
    unsafe { sort_values_and_permute_with_bit_reverse_order_cuda(from, size) }
}

pub unsafe fn precompute_twiddles(
    initial: CirclePointBaseField,
    step: CirclePointBaseField,
    total_size: usize,
) -> *const u32 {
    let _ = span!(
        Level::INFO,
        "precompute_twiddles",
        message = format!("called with total_size: {:?}", total_size)
    )
    .entered();
    unsafe { precompute_twiddles_cuda(initial, step, total_size) }
}

pub unsafe fn interpolate(
    eval_domain_size: u32,
    values: *const u32,
    inverse_twiddles_tree: *const u32,
    inverse_twiddle_tree_size: u32,
    values_size: u32,
) {
    let _ = span!(Level::INFO,
        "iNTT interpolate", message = format!("called with eval_domain_size: {:?}, inverse_twiddle_tree_size: {:?}, values_size: {:?}",
        eval_domain_size, inverse_twiddle_tree_size, values_size)
    ).entered();
    unsafe {
        interpolate_cuda(
            eval_domain_size,
            values,
            inverse_twiddles_tree,
            inverse_twiddle_tree_size,
            values_size,
        );
    }
}

pub unsafe fn evaluate(
    eval_domain_size: u32,
    values: *const u32,
    twiddles_tree: *const u32,
    twiddle_tree_size: u32,
    values_size: u32,
) {
    let _ = span!(
        Level::INFO,
        "NTT evaluate",
        message = format!(
            "called with eval_domain_size: {:?}, twiddle_tree_size: {:?}, values_size: {:?}",
            eval_domain_size, twiddle_tree_size, values_size
        )
    )
    .entered();
    unsafe {
        evaluate_cuda(
            eval_domain_size,
            values,
            twiddles_tree,
            twiddle_tree_size,
            values_size,
        );
    }
}

pub unsafe fn eval_at_point(
    coeffs: *const u32,
    coeffs_size: u32,
    point_x: CudaSecureField,
    point_y: CudaSecureField,
) -> CudaSecureField {
    let _ = span!(
        Level::INFO,
        "eval_at_point",
        message = format!("called with coeffs_size: {:?}", coeffs_size)
    )
    .entered();
    unsafe { eval_at_point_cuda(coeffs, coeffs_size, point_x, point_y) }
}

pub unsafe fn fold_line(
    gpu_domain: *const u32,
    twiddle_offset: usize,
    n: usize,
    eval_values: *const *const u32,
    alpha: CudaSecureField,
    folded_values: *const *const u32,
) {
    let _ = span!(
        Level::INFO,
        "fold_line",
        message = format!(
            "called with twiddle_offset: {:?}, n: {:?}",
            twiddle_offset, n
        )
    )
    .entered();
    unsafe {
        fold_line_cuda(
            gpu_domain,
            twiddle_offset,
            n,
            eval_values,
            alpha,
            folded_values,
        );
    }
}

pub unsafe fn fold_circle_into_line(
    gpu_domain: *const u32,
    twiddle_offset: usize,
    n: usize,
    eval_values: *const *const u32,
    alpha: CudaSecureField,
    folded_values: *const *const u32,
) {
    let _ = span!(
        Level::INFO,
        "fold_circle_into_line",
        message = format!(
            "called with twiddle_offset: {:?}, n: {:?}",
            twiddle_offset, n
        )
    )
    .entered();
    unsafe {
        fold_circle_into_line_cuda(
            gpu_domain,
            twiddle_offset,
            n,
            eval_values,
            alpha,
            folded_values,
        );
    }
}

pub unsafe fn decompose(
    columns: *const *const u32,
    column_size: u32,
    lambda: &CudaSecureField,
    g_values: *const *const u32,
) {
    let _ = span!(
        Level::INFO,
        "decompose",
        message = format!("called with column_size: {:?}", column_size)
    )
    .entered();
    unsafe { decompose_cuda(columns, column_size, lambda, g_values) }
}

pub unsafe fn accumulate(
    size: u32,
    left_columns: *const *const u32,
    right_columns: *const *const u32,
) {
    let _ = span!(
        Level::INFO,
        "accumulate",
        message = format!("called with size: {:?}", size)
    )
    .entered();
    unsafe { accumulate_cuda(size, left_columns, right_columns) }
}

pub unsafe fn commit_on_first_layer(
    size: usize,
    amount_of_columns: usize,
    columns: *const *const u32,
    result: *mut Blake2sHash,
) {
    let _ = span!(
        Level::INFO,
        "commit_on_first_layer",
        message = format!(
            "called with size: {:?}, amount_of_columns: {:?}",
            size, amount_of_columns
        )
    )
    .entered();
    unsafe { commit_on_first_layer_cuda(size, amount_of_columns, columns, result) }
}

pub unsafe fn commit_on_layer_with_previous(
    size: usize,
    amount_of_columns: usize,
    columns: *const *const u32,
    previous_layer: *const Blake2sHash,
    result: *mut Blake2sHash,
) {
    let _ = span!(
        Level::INFO,
        "commit_on_layer_with_previous",
        message = format!(
            "called with size: {:?}, amount_of_columns: {:?}",
            size, amount_of_columns
        )
    )
    .entered();
    unsafe {
        commit_on_layer_with_previous_cuda(
            size,
            amount_of_columns,
            columns,
            previous_layer,
            result,
        );
    }
}

pub unsafe fn copy_blake_2s_hash_vec_from_host_to_device(
    from: *const Blake2sHash,
    size: usize,
) -> *mut Blake2sHash {
    info!(
        target = " ==TO DEVICE>>: copy_blake_2s_hash_vec_from_host_to_device",
        message = format!("called with size: {:?}", size)
    );
    unsafe { copy_blake_2s_hash_vec_from_host_to_device_cuda(from, size) }
}

pub unsafe fn copy_blake_2s_hash_vec_from_device_to_host(
    from: *const Blake2sHash,
    to: *const Blake2sHash,
    size: usize,
) {
    info!(
        target = " <<<TO HOST==: copy_blake_2s_hash_vec_from_device_to_host",
        message = format!("called with size: {:?}", size)
    );
    unsafe { copy_blake_2s_hash_vec_from_device_to_host_cuda(from, to, size) }
}

pub unsafe fn copy_blake_2s_hash_vec_from_device_to_device(
    from: *const Blake2sHash,
    dst: *const Blake2sHash,
    size: usize,
) {
    info!(
        target = "==DtoD==: copy_blake_2s_hash_vec_from_device_to_device",
        message = format!("called with size: {:?}", size)
    );
    unsafe { copy_blake_2s_hash_vec_from_device_to_device_cuda(from, dst, size) }
}

pub unsafe fn free_blake_2s_hash_vec(device_pointer: *const Blake2sHash) {
    info!("free_blake_2s_hash_vec called");
    unsafe { free_blake_2s_hash_vec_cuda(device_pointer) }
}

pub unsafe fn copy_device_pointer_vec_from_host_to_device(
    from: *const *const u32,
    size: usize,
) -> *const *const u32 {
    info!(
        target = " ==TO DEVICE>>: copy_device_pointer_vec_from_host_to_device",
        message = format!("called with size: {:?}", size)
    );
    unsafe { copy_device_pointer_vec_from_host_to_device_cuda(from, size) }
}

pub unsafe fn free_device_pointer_vec(device_pointer: *const *const u32) {
    info!("free_device_pointer_vec called");
    unsafe { free_device_pointer_vec_cuda(device_pointer) }
}

pub unsafe fn accumulate_quotients(
    half_coset_initial_index: u32,
    half_coset_step_size: u32,
    domain_size: u32,
    columns: *const *const u32,
    number_of_columns: usize,
    random_coeff: CudaSecureField,
    sample_points: *const u32,
    sample_columns_indexes: *const u32,
    sample_columns_indexes_size: u32,
    sample_column_values: *const CudaSecureField,
    sample_column_and_values_sizes: *const u32,
    sample_size: u32,
    result_column_0: *const u32,
    result_column_1: *const u32,
    result_column_2: *const u32,
    result_column_3: *const u32,
    flattened_line_coeffs_size: u32,
) {
    let _ = span!(
        Level::INFO,
        "accumulate_quotients",
        message = format!(
            "called with domain_size: {:?}, number_of_columns: {:?}, sample_size: {:?}",
            domain_size, number_of_columns, sample_size
        )
    )
    .entered();
    unsafe {
        accumulate_quotients_cuda(
            half_coset_initial_index,
            half_coset_step_size,
            domain_size,
            columns,
            number_of_columns,
            random_coeff,
            sample_points,
            sample_columns_indexes,
            sample_columns_indexes_size,
            sample_column_values,
            sample_column_and_values_sizes,
            sample_size,
            result_column_0,
            result_column_1,
            result_column_2,
            result_column_3,
            flattened_line_coeffs_size,
        );
    }
}

pub unsafe fn fibonacci_component_evaluate_constraint_quotients_on_domain(
    evals: *const u32,
    evals_size: u32,
    output_column_0: *const u32,
    output_column_1: *const u32,
    output_column_2: *const u32,
    output_column_3: *const u32,
    claim_value: BaseField,
    initial_point: CirclePointBaseField,
    step_point: CirclePointBaseField,
    random_coeff_0: CudaSecureField,
    random_coeff_1: CudaSecureField,
) {
    let _ = span!(
        Level::INFO,
        "fibonacci_component_evaluate_constraint_quotients_on_domain",
        message = format!("called with evals_size: {:?}", evals_size)
    )
    .entered();
    unsafe {
        fibonacci_component_evaluate_constraint_quotients_on_domain_cuda(
            evals,
            evals_size,
            output_column_0,
            output_column_1,
            output_column_2,
            output_column_3,
            claim_value,
            initial_point,
            step_point,
            random_coeff_0,
            random_coeff_1,
        );
    }
}
