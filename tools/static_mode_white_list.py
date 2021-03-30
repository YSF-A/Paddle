# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

STATIC_MODE_TESTING_LIST = [
    'test_affine_channel_op',
    'test_concat_op',
    'test_elementwise_add_op',
    'test_elementwise_sub_op',
    'test_fill_zeros_like2_op',
    'test_linear_chain_crf_op',
    'test_lod_reset_op',
    'test_lookup_table_op',
    'test_lookup_table_bf16_op',
    'test_pad2d_op',
    'test_scatter_op',
    'test_sequence_concat',
    'test_sequence_conv',
    'test_sequence_pool',
    'test_sequence_expand_as',
    'test_sequence_expand',
    'test_sequence_pad_op',
    'test_sequence_unpad_op',
    'test_sequence_scatter_op',
    'test_sequence_slice_op',
    'test_slice_op',
    'test_space_to_depth_op',
    'test_squared_l2_distance_op',
    'test_accuracy_op',
    'test_activation_nn_grad',
    'test_adadelta_op',
    'test_adagrad_op',
    'test_adam_op',
    'test_adam_optimizer_fp32_fp64',
    'test_adamax_api',
    'test_adamax_op',
    'test_adamw_op',
    'test_adaptive_avg_pool1d',
    'test_adaptive_max_pool1d',
    'test_add_position_encoding_op',
    'test_add_reader_dependency',
    'test_addmm_op',
    'test_affine_grid_op',
    'test_allclose_layer',
    'test_amp_check_finite_and_scale_op',
    'test_anchor_generator_op',
    'test_arange',
    'test_arg_min_max_op',
    'test_argsort_op',
    'test_array_read_write_op',
    'test_assert_op',
    'test_assign_op',
    'test_assign_value_op',
    'test_attention_lstm_op',
    'test_auc_op',
    'test_auc_single_pred_op',
    'test_avoid_twice_initialization',
    'test_backward',
    'test_basic_rnn_name',
    'test_batch_norm_op',
    'test_batch_norm_op_v2',
    'test_bce_loss',
    'test_beam_search_decode_op',
    'test_beam_search_op',
    'test_bicubic_interp_op',
    'test_bicubic_interp_v2_op',
    'test_bilateral_slice_op',
    'test_bilinear_api',
    'test_bilinear_interp_v2_op',
    'test_bilinear_tensor_product_op',
    'test_bipartite_match_op',
    'test_bmm_op',
    'test_box_clip_op',
    'test_box_coder_op',
    'test_box_decoder_and_assign_op',
    'test_bpr_loss_op',
    'test_calc_gradient',
    'test_case',
    'test_cast_op',
    'test_center_loss',
    'test_cholesky_op',
    'test_chunk_eval_op',
    'test_chunk_op',
    'test_clip_by_norm_op',
    'test_clip_op',
    'test_collect_fpn_proposals_op',
    'test_compare_reduce_op',
    'test_compiled_program',
    'test_cond',
    'test_conditional_block',
    'test_context_manager',
    'test_conv1d_layer',
    'test_conv1d_transpose_layer',
    'test_conv2d_layer',
    'test_conv2d_op',
    'test_conv2d_transpose_layer',
    'test_conv3d_layer',
    'test_conv3d_op',
    'test_conv3d_transpose_layer',
    'test_conv3d_transpose_part2_op',
    'test_conv_nn_grad',
    'test_conv_transpose_nn_grad',
    'test_conv_shift_op',
    'test_cos_sim_op',
    'test_create_global_var',
    'test_crf_decoding_op',
    'test_crop_op',
    'test_crop_tensor_op',
    'test_cross_entropy2_op',
    'test_cross_entropy_loss',
    'test_cross_entropy_op',
    'test_cross_op',
    'test_ctc_align',
    'test_cumsum_op',
    'test_cvm_op',
    'test_data',
    'test_dataloader_early_reset',
    'test_dataloader_keep_order',
    'test_dataloader_unkeep_order',
    'test_debugger',
    'test_decayed_adagrad_op',
    'test_decoupled_py_reader',
    'test_decoupled_py_reader_data_check',
    'test_deformable_conv_v1_op',
    'test_deformable_psroi_pooling',
    'test_density_prior_box_op',
    'test_deprecated_memory_optimize_interfaces',
    'test_dequantize_abs_max_op',
    'test_dequantize_log_op',
    'test_desc_clone',
    'test_detach',
    'test_device',
    'test_device_guard',
    'test_diag_embed',
    'test_distribute_fpn_proposals_op',
    'test_distributed_strategy',
    'test_distributions',
    'test_dot_op',
    'test_downpoursgd',
    'test_dpsgd_op',
    'test_dropout_op',
    'test_dygraph_multi_forward',
    'test_dyn_rnn',
    'test_dynamic_rnn_stop_gradient',
    'test_dynrnn_gradient_check',
    'test_dynrnn_static_input',
    'test_eager_deletion_conditional_block',
    'test_eager_deletion_delete_vars',
    'test_eager_deletion_gru_net',
    'test_eager_deletion_lstm_net',
    'test_eager_deletion_padding_rnn',
    'test_eager_deletion_recurrent_op',
    'test_eager_deletion_while_op',
    'test_edit_distance_op',
    'test_elementwise_div_op',
    'test_elementwise_floordiv_op',
    'test_elementwise_gradient_op',
    'test_elementwise_max_op',
    'test_elementwise_min_op',
    'test_elementwise_mod_op',
    'test_elementwise_mul_op',
    'test_elementwise_nn_grad',
    'test_elementwise_pow_op',
    'test_ema',
    'test_embedding_id_stop_gradient',
    'test_empty_like_op',
    'test_entry_attr',
    'test_entry_attr2',
    'test_erf_op',
    'test_executor_and_mul',
    'test_executor_and_use_program_cache',
    'test_executor_check_feed',
    'test_executor_feed_non_tensor',
    'test_executor_return_tensor_not_overwriting',
    'test_expand_as_op',
    'test_expand_as_v2_op',
    'test_expand_op',
    'test_expand_v2_op',
    'test_eye_op',
    'test_fake_dequantize_op',
    'test_fake_quantize_op',
    'test_fc_op',
    'test_feed_data_check_shape_type',
    'test_fetch_lod_tensor_array',
    'test_fetch_unmerged',
    'test_fetch_var',
    'test_fill_any_like_op',
    'test_fill_constant_op',
    'test_fill_op',
    'test_fill_zeros_like_op',
    'test_filter_by_instag_op',
    'test_flatten2_op',
    'test_flatten_contiguous_range_op',
    'test_flatten_op',
    'test_fleet',
    'test_fleet_nocvm_1',
    'test_fleet_pyramid_hash',
    'test_fleet_rolemaker',
    'test_fleet_rolemaker_3',
    'test_fleet_unitaccessor',
    'test_fleet_util',
    'test_fleet_utils',
    'test_flip',
    'test_framework_debug_str',
    'test_fsp_op',
    'test_ftrl_op',
    'test_full_like_op',
    'test_full_op',
    'test_functional_conv2d',
    'test_functional_conv2d_transpose',
    'test_functional_conv3d',
    'test_functional_conv3d_transpose',
    'test_fuse_all_reduce_pass',
    'test_fuse_optimizer_pass',
    'test_fuse_relu_depthwise_conv_pass',
    'test_fused_elemwise_activation_op',
    'test_fused_emb_seq_pool_op',
    'test_fused_embedding_fc_lstm_op',
    'test_fusion_gru_op',
    'test_fusion_lstm_op',
    'test_fusion_repeated_fc_relu_op',
    'test_fusion_seqconv_eltadd_relu_op',
    'test_fusion_seqpool_concat_op',
    'test_fusion_seqpool_cvm_concat_op',
    'test_fusion_squared_mat_sub_op',
    'test_gather_tree_op',
    'test_gaussian_random_op',
    'test_generate_mask_labels_op',
    'test_generate_proposal_labels_op',
    'test_generate_proposals_op',
    'test_generator_dataloader',
    'test_get_places_op',
    'test_get_tensor_from_selected_rows_op',
    'test_gradient_clip',
    'test_grid_sample_function',
    'test_grid_sampler_op',
    'test_group_norm_op',
    'test_group_norm_op_v2',
    'test_gru_op',
    'test_gru_unit_op',
    'test_hash_op',
    'test_hinge_loss_op',
    'test_histogram_op',
    'test_huber_loss_op',
    'test_im2sequence_op',
    'test_image_classification_layer',
    'test_imperative_basic',
    'test_imperative_deepcf',
    'test_imperative_framework',
    'test_imperative_gan',
    'test_imperative_gnn',
    'test_imperative_load_static_param',
    'test_imperative_lod_tensor_to_selected_rows',
    'test_imperative_optimizer',
    'test_imperative_ptb_rnn',
    'test_imperative_ptb_rnn_sorted_gradient',
    'test_imperative_recurrent_usage',
    'test_imperative_reinforcement',
    'test_imperative_selected_rows_to_lod_tensor',
    'test_imperative_star_gan_with_gradient_penalty',
    'test_imperative_transformer_sorted_gradient',
    'test_increment',
    'test_index_sample_op',
    'test_index_select_op',
    'test_infer_no_need_buffer_slots',
    'test_inference_model_io',
    'test_initializer',
    'test_inplace_abn_op',
    'test_inplace_addto_strategy',
    'test_inplace_softmax_with_cross_entropy',
    'test_input_spec',
    'test_instance_norm_op',
    'test_instance_norm_op_v2',
    'test_inverse_op',
    'test_io_save_load',
    'test_iou_similarity_op',
    'test_ir_memory_optimize_ifelse_op',
    'test_ir_memory_optimize_pass',
    'test_is_empty_op',
    'test_isfinite_op',
    'test_kldiv_loss_op',
    'test_kron_op',
    'test_l1_norm_op',
    'test_label_smooth_op',
    'test_lamb_op',
    'test_layer_norm_op',
    'test_layer_norm_mkldnn_op',
    'test_layer_norm_bf16_mkldnn_op',
    'test_layer_norm_op_v2',
    'test_layer_norm_fuse_pass',
    'test_learning_rate_scheduler',
    'test_linear_interp_op',
    'test_linear_interp_v2_op',
    'test_linspace',
    'test_load_op',
    'test_load_vars_shape_check',
    'test_locality_aware_nms_op',
    'test_lod_append_op',
    'test_lod_array_length_op',
    'test_lod_rank_table',
    'test_lod_tensor_array_ops',
    'test_log_loss_op',
    'test_log_softmax',
    'test_logsumexp',
    'test_lookup_table_dequant_op',
    'test_lookup_table_v2_op',
    'test_lrn_op',
    'test_lstm_op',
    'test_lstmp_op',
    'test_margin_rank_loss_op',
    'test_math_op_patch',
    'test_matmul_op',
    'test_matmul_v2_op',
    'test_matrix_nms_op',
    'test_mean_iou',
    'test_memory_reuse_exclude_feed_var',
    'test_memory_usage',
    'test_merge_ids_op',
    'test_meshgrid_op',
    'test_mine_hard_examples_op',
    'test_minus_op',
    'test_mish_op',
    'test_modified_huber_loss_op',
    'test_momentum_op',
    'test_monitor',
    'test_mse_loss',
    'test_mul_op',
    'test_multiclass_nms_op',
    'test_multihead_attention',
    'test_multiplex_op',
    'test_multiprocess_reader_exception',
    'test_name_scope',
    'test_nce',
    'test_nearest_interp_v2_op',
    'test_network_with_dtype',
    'test_nll_loss',
    'test_nn_functional_embedding_static',
    'test_nn_functional_hot_op',
    'test_nonzero_api',
    'test_norm_all',
    'test_norm_nn_grad',
    'test_norm_op',
    'test_normal',
    'test_normalization_wrapper',
    'test_npair_loss_op',
    'test_numel_op',
    'test_one_hot_op',
    'test_one_hot_v2_op',
    'test_ones_like',
    'test_ones_op',
    'test_op_name_conflict',
    'test_operator_desc',
    'test_optimizer',
    'test_optimizer_in_control_flow',
    'test_pad_constant_like',
    'test_pad_op',
    'test_pairwise_distance',
    'test_parallel_executor_drop_scope',
    'test_parallel_executor_dry_run',
    'test_parallel_executor_feed_persistable_var',
    'test_parallel_executor_inference_feed_partial_data',
    'test_parallel_executor_mnist',
    'test_parallel_executor_run_load_infer_program',
    'test_parallel_executor_test_while_train',
    'test_parallel_ssa_graph_inference_feed_partial_data',
    'test_parameter',
    'test_partial_concat_op',
    'test_partial_eager_deletion_transformer',
    'test_partial_sum_op',
    'test_pass_builder',
    'test_pixel_shuffle',
    'test_polygon_box_transform',
    'test_pool1d_api',
    'test_pool2d_api',
    'test_pool2d_op',
    'test_pool3d_api',
    'test_pool3d_op',
    'test_pool_max_op',
    'test_positive_negative_pair_op',
    'test_precision_recall_op',
    'test_prelu_op',
    'test_print_op',
    'test_prior_box_op',
    'test_profiler',
    'test_program',
    'test_program_code',
    'test_program_prune_backward',
    'test_program_to_string',
    'test_protobuf_descs',
    'test_proximal_adagrad_op',
    'test_proximal_gd_op',
    'test_prroi_pool_op',
    'test_prune',
    'test_psroi_pool_op',
    'test_py_func_op',
    'test_py_reader_combination',
    'test_py_reader_lod_level_share',
    'test_py_reader_pin_memory',
    'test_py_reader_push_pop',
    'test_py_reader_return_list',
    'test_py_reader_sample_generator',
    'test_py_reader_using_executor',
    'test_pyramid_hash_op',
    'test_queue',
    'test_randint_op',
    'test_randn_op',
    'test_random_crop_op',
    'test_randperm_op',
    'test_range',
    'test_rank_loss_op',
    'test_reader_reset',
    'test_recurrent_op',
    'test_reduce_op',
    'test_ref_by_trainer_id_op',
    'test_registry',
    'test_regularizer',
    'test_regularizer_api',
    'test_reorder_lod_tensor',
    'test_reshape_op',
    'test_reshape_bf16_op',
    'test_retinanet_detection_output',
    'test_reverse_op',
    'test_rmsprop_op',
    'test_rnn_cell_api',
    'test_rnn_memory_helper_op',
    'test_roi_align_op',
    'test_roi_perspective_transform_op',
    'test_roi_pool_op',
    'test_roll_op',
    'test_row_conv',
    'test_row_conv_op',
    'test_rpn_target_assign_op',
    'test_run_program_op',
    'test_runtime_and_compiletime_exception',
    'test_sample_logits_op',
    'test_save_model_without_var',
    'test_scale_op',
    'test_scaled_dot_product_attention',
    'test_scatter_nd_op',
    'test_seed_op',
    'test_segment_ops',
    'test_select_input_output_op',
    'test_selu_op',
    'test_set_bool_attr',
    'test_sgd_op',
    'test_shape_op',
    'test_shard_index_op',
    'test_shrink_rnn_memory',
    'test_shuffle_batch_op',
    'test_shuffle_channel_op',
    'test_sigmoid_cross_entropy_with_logits_op',
    'test_sigmoid_focal_loss_op',
    'test_sign_op',
    'test_similarity_focus_op',
    'test_size_op',
    'test_smooth_l1_loss',
    'test_smooth_l1_loss_op',
    'test_softmax_with_cross_entropy_op',
    'test_spectral_norm_op',
    'test_split_and_merge_lod_tensor_op',
    'test_split_ids_op',
    'test_split_op',
    'test_spp_op',
    'test_square_error_cost',
    'test_squared_l2_norm_op',
    'test_stack_op',
    'test_static_save_load',
    'test_sum_op',
    'test_switch',
    'test_switch_case',
    'test_target_assign_op',
    'test_tdm_child_op',
    'test_tdm_sampler_op',
    'test_teacher_student_sigmoid_loss_op',
    'test_temporal_shift_op',
    'test_tensor_array_to_tensor',
    'test_tile_op',
    'test_top_k_op',
    'test_trace_op',
    'test_trainable',
    'test_transpose_op',
    'test_tree_conv_op',
    'test_tril_triu_op',
    'test_trilinear_interp_op',
    'test_trilinear_interp_v2_op',
    'test_truncated_gaussian_random_op',
    'test_unbind_op',
    'test_unfold_op',
    'test_uniform_random_op',
    'test_unique',
    'test_unique_with_counts',
    'test_unpool_op',
    'test_unstack_op',
    'test_update_loss_scaling_op',
    'test_var_info',
    'test_variable',
    'test_weight_normalization',
    'test_where_index',
    'test_where_op',
    'test_yolo_box_op',
    'test_yolov3_loss_op',
    'test_zeros_like_op',
    'test_zeros_op',
    'test_adam_op_multi_thread',
    'test_bilinear_interp_op',
    'test_nearest_interp_op',
    'test_imperative_resnet',
    'test_imperative_resnet_sorted_gradient',
    'test_imperative_mnist',
    'test_imperative_mnist_sorted_gradient',
    'test_imperative_se_resnext',
    'test_imperative_ocr_attention_model',
    'test_imperative_static_runner_mnist',
    'test_imperative_static_runner_while',
    'test_recv_save_op',
    'test_transpiler_ops',
    'test_communicator_sync',
    'test_collective_optimizer',
    'test_parallel_executor_crf',
    'test_parallel_executor_profiler',
    'test_parallel_executor_transformer',
    'test_parallel_executor_transformer_auto_growth',
    'test_data_norm_op',
    'test_fuse_bn_act_pass',
    'test_parallel_executor_seresnext_base_cpu',
    'test_parallel_executor_seresnext_with_reduce_cpu',
    'test_parallel_executor_seresnext_with_fuse_all_reduce_cpu',
    'test_layers',
    'test_parallel_executor_fetch_feed',
    'test_sequence_concat',
    'test_sequence_conv',
    'test_sequence_enumerate_op',
    'test_sequence_erase_op',
    'test_sequence_expand',
    'test_sequence_expand_as',
    'test_sequence_first_step',
    'test_sequence_last_step',
    'test_sequence_mask',
    'test_sequence_pad_op',
    'test_sequence_pool',
    'test_sequence_reshape',
    'test_sequence_reverse',
    'test_sequence_scatter_op',
    'test_sequence_slice_op',
    'test_sequence_softmax_op',
    'test_sequence_topk_avg_pooling',
    'test_sequence_unpad_op',
    'test_ast_util',
    'test_basic_api_transformation',
    'test_function_spec',
    'test_len',
    'test_slice',
    'test_variable_trans_func',
    'test_ir_embedding_eltwise_layernorm_fuse_pass',
    'test_ir_fc_fuse_pass',
    'test_ir_skip_layernorm_pass',
    'test_conv_affine_channel_fuse_pass',
    'test_conv_bias_mkldnn_fuse_pass',
    'test_conv_bn_fuse_pass',
    'test_conv_elementwise_add2_act_fuse_pass',
    'test_conv_elementwise_add_act_fuse_pass',
    'test_conv_elementwise_add_fuse_pass',
    'test_fc_fuse_pass',
    'test_fc_gru_fuse_pass',
    'test_fc_lstm_fuse_pass',
    'test_repeated_fc_relu_fuse_pass',
    'test_seqconv_eltadd_relu_fuse_pass',
    'test_squared_mat_sub_fuse_pass',
    'test_transpose_flatten_concat_fuse_pass',
    'test_detection_map_op',
    'test_fuse_elewise_add_act_pass',
    'test_fusion_seqexpand_concat_fc_op',
    'test_match_matrix_tensor_op',
    'test_matmul_op_with_head',
    'test_var_conv_2d',
    'test_batch_norm_mkldnn_op',
    'test_concat_int8_mkldnn_op',
    'test_concat_bf16_mkldnn_op',
    'test_concat_mkldnn_op',
    'test_conv2d_bf16_mkldnn_op',
    'test_conv2d_int8_mkldnn_op',
    'test_conv2d_mkldnn_op',
    'test_conv2d_transpose_mkldnn_op',
    'test_conv2d_transpose_bf16_mkldnn_op',
    'test_conv3d_mkldnn_op',
    'test_dequantize_mkldnn_op',
    'test_elementwise_add_mkldnn_op',
    'test_elementwise_add_bf16_mkldnn_op',
    'test_elementwise_mul_mkldnn_op',
    'test_elementwise_mul_bf16_mkldnn_op',
    'test_fc_mkldnn_op',
    'test_fc_bf16_mkldnn_op',
    'test_nearest_interp_mkldnn_op',
    'test_bilinear_interp_mkldnn_op',
    'test_fusion_gru_int8_mkldnn_op',
    'test_fusion_gru_bf16_mkldnn_op',
    'test_fusion_gru_mkldnn_op',
    'test_fusion_lstm_mkldnn_op',
    'test_fusion_lstm_int8_mkldnn_op',
    'test_fusion_lstm_bf16_mkldnn_op',
    'test_gaussian_random_mkldnn_op',
    'test_lrn_mkldnn_op',
    'test_matmul_mkldnn_op',
    'test_matmul_bf16_mkldnn_op',
    'test_mul_int8_mkldnn_op',
    'test_multi_gru_mkldnn_op',
    'test_multi_gru_fuse_pass',
    'test_multi_gru_seq_fuse_pass',
    'test_pool2d_int8_mkldnn_op',
    'test_pool2d_bf16_mkldnn_op',
    'test_pool2d_mkldnn_op',
    'test_quantize_mkldnn_op',
    'test_requantize_mkldnn_op',
    'test_softmax_mkldnn_op',
    'test_softmax_bf16_mkldnn_op',
    'test_sum_mkldnn_op',
    'test_sum_bf16_mkldnn_op',
    'test_transpose_int8_mkldnn_op',
    'test_transpose_bf16_mkldnn_op',
    'test_transpose_mkldnn_op',
    'test_mkldnn_conv_activation_fuse_pass',
    'test_mkldnn_conv_concat_relu_mkldnn_fuse_pass',
    'test_mkldnn_matmul_op_output_fuse_pass',
    'test_mkldnn_matmul_transpose_reshape_fuse_pass',
    'test_mkldnn_scale_matmul_fuse_pass',
    'test_mkldnn_inplace_fuse_pass',
    'test_batch_fc_op',
    'test_c_comm_init_all_op',
    'test_conv2d_fusion_op',
    'test_dataset_dataloader',
    'test_fleet_metric',
    'test_fused_bn_add_act',
    'test_fused_multihead_matmul_op',
    'test_ir_inplace_pass',
    'test_mix_precision_all_reduce_fuse',
    'test_parallel_executor_pg',
    'test_rank_attention_op',
    'test_fleet_base',
    'test_fleet_graph_executor',
    'test_fleet_meta_optimizer_base',
    'test_ir_memory_optimize_transformer',
    'test_trt_fc_fuse_pass',
    'test_trt_quant_conv2d_dequant_fuse_pass',
    'test_trt_slice_plugin',
    'test_trt_transpose_flatten_concat_fuse_pass',
    'test_mean_op',
    'test_build_strategy_fusion_group_pass',
    'test_coalesce_tensor_op',
    'test_dataset',
    'test_fleet_base_single',
    'test_fleet_rolemaker_new',
    'test_fused_fc_elementwise_layernorm_op',
    'test_fusion_transpose_flatten_concat_op',
    'test_ir_memory_optimize_nlp',
    'test_nvprof',
    'test_pipeline',
    'test_weight_decay',
    'test_fleet_base_2',
    'test_fleet_pipeline_meta_optimizer',
    'test_fleet_checkpoint',
    'test_ir_fusion_group_pass',
    'test_trt_pad_op',
    'test_trt_shuffle_channel_detect_pass',
    'test_trt_subgraph_pass',
    'test_parallel_executor_seresnext_base_gpu',
    'test_parallel_executor_seresnext_with_fuse_all_reduce_gpu',
    'test_parallel_executor_seresnext_with_reduce_gpu',
    'test_sync_batch_norm_op',
    'test_multiprocess_dataloader_iterable_dataset_static',
    'test_multiprocess_dataloader_static',
    'test_load_op_xpu',
    'test_activation_op_xpu',
    'test_adam_op_xpu',
    'test_assign_op_xpu',
    'test_batch_norm_op_xpu',
    'test_cast_op_xpu',
    'test_concat_op_xpu',
    'test_elementwise_add_op_xpu',
    'test_fill_constant_op_xpu',
    'test_gather_op_xpu',
    'test_matmul_op_xpu',
    'test_matmul_v2_op_xpu',
    'test_mean_op_xpu',
    'test_momentum_op_xpu',
    'test_reduce_mean_op_xpu',
    'test_reduce_sum_op_xpu',
    'test_reshape2_op_xpu',
    'test_sgd_op_xpu',
    'test_shape_op_xpu',
    'test_slice_op_xpu',
    'test_generate_proposals_v2_op',
    'test_lamb_op_xpu',
    'test_model_cast_to_bf16',
]
