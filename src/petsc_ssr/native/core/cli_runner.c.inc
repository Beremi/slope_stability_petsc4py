PetscErrorCode P4IndirectSSRRunFromOptions(void)
{
  AppCtx          app;
  P4Basis         basis;
  DM              dm = NULL;
  AssemblyCtx     actx;
  LinearSolverCtx lsolver;
  SSRStats        ssr_stats;
  Mat             Areg = NULL, Kelastic = NULL, Ktangent = NULL;
  Vec             u = NULL, f_ext = NULL;
  PetscInt        cStart, cEnd, nStart, nEnd;
  PetscInt        local_dofs, global_dofs, reported_owned_constraints = 0, reported_global_constraints = 0;
  PetscReal       rhs_norm;
  PetscLogDouble  elastic_assembly_time, dirichlet_time, wall_time;
  SsrProfileTimer run_timer, profile_timer;

  PetscFunctionBeginUser;
  PetscCall(RegisterLogStages());
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_ENGINE_RUN, NULL, NULL, &run_timer);
  PetscCall(ParseOptions(PETSC_COMM_WORLD, &app));
  PetscCall(NativeProblemManifestApply(&app));
  PetscCall(MaterialMCConfigureFromOptions(PETSC_COMM_WORLD));
  PetscCall(SetDDPartitionerDefault(PETSC_COMM_WORLD, &app));
  PetscCall(P4BasisCreateDegreeDim(PETSC_COMM_SELF, app.element_degree, app.mechanics_dim, app.mechanics_dim, &basis));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &app, &basis, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &nStart, &nEnd));
  PetscCall(DMCreateMatrix(dm, &Areg));
  PetscCall(DMCreateMatrix(dm, &Kelastic));
  PetscCall(DMCreateMatrix(dm, &Ktangent));
  PetscCall(MatSetOption(Areg, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(MatSetOption(Kelastic, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(MatSetOption(Ktangent, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(DMCreateGlobalVector(dm, &f_ext));
  PetscCall(VecGetLocalSize(u, &local_dofs));
  PetscCall(VecGetSize(u, &global_dofs));
  PetscCall(AssemblyCtxCreate(dm, &basis, &actx));
  reported_owned_constraints = actx.n_constrained_local;
  reported_global_constraints = actx.n_constrained_all;
  {
    PetscBool                    label_constraints_loaded = PETSC_FALSE;
    AssemblyLabelConstraintStats label_stats;

    PetscCall(AssemblyCtxLoadLabelConstraintsCSV(&actx, app.mechanics_bc_labels_csv, &label_constraints_loaded, &label_stats));
    if (label_constraints_loaded) {
      reported_owned_constraints = label_stats.owned_constraints;
      reported_global_constraints = label_stats.global_constraints;
    }
    if (app.native_problem_manifest_loaded) {
      PetscCheck(label_stats.rows == app.native_manifest_mechanics_dirichlet, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
                 "Native problem manifest declares %" PetscInt_FMT " mechanics Dirichlet row(s), but label table contains %" PetscInt_FMT,
                 app.native_manifest_mechanics_dirichlet, label_stats.rows);
    }
    PetscCheck(!app.mechanics_bc_nodes_csv[0] || app.debug_coordinate_bc_table, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "-mechanics_bc_nodes_csv requires -debug_coordinate_bc_table true");
    if (app.mechanics_bc_nodes_csv[0]) {
      if (label_constraints_loaded) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MECHANICS_BC_NODES_CONFIG path=%s status=debug_coordinate_override\n", app.mechanics_bc_nodes_csv));
      PetscCall(AssemblyCtxLoadCoordinateConstraintsCSV(&actx, app.mechanics_bc_nodes_csv));
      reported_owned_constraints = actx.n_constrained_local;
      reported_global_constraints = actx.n_constrained_all;
    } else if (!label_constraints_loaded) PetscCall(AssemblyCtxLoadCoordinateConstraintsCSV(&actx, app.mechanics_bc_nodes_csv));
  }
  if (app.seepage_pressure_csv[0]) {
    PetscBool pressure_source_ok = PETSC_FALSE;

    PetscCall(PetscStrcasecmp(app.seepage_pressure_source, "hydro_prepass_coordinate_bridge", &pressure_source_ok));
    PetscCheck(pressure_source_ok, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "-seepage_pressure_csv requires -seepage_pressure_source hydro_prepass_coordinate_bridge");
  }
  PetscCall(AssemblyCtxLoadSeepagePressureCSV(&actx, app.seepage_pressure_csv, app.seepage_grho));
  PetscCall(AssemblyCtxValidateSeepageBoundaryLabelsCSV(&actx, app.seepage_boundary_labels_csv,
                                                        app.native_problem_manifest_loaded ? app.native_manifest_seepage_head : -1,
                                                        app.native_problem_manifest_loaded ? app.native_manifest_seepage_flux : -1));
  PetscCall(AssemblyCtxValidateNeumannLabelsCSV(&actx, app.mechanics_neumann_labels_csv,
                                                app.native_problem_manifest_loaded ? app.native_manifest_mechanics_neumann : -1));
  if (app.native_problem_manifest[0]) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "NATIVE_PROBLEM_MANIFEST path=%s\n", app.native_problem_manifest));
  PetscCall(LinearSolverInit(&lsolver, dm, &actx, &app, Areg));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "mesh=%s%s refine_levels=%" PetscInt_FMT " mechanics_dim=%" PetscInt_FMT " element_degree=%" PetscInt_FMT " local_cells=%" PetscInt_FMT " local_vertices=%" PetscInt_FMT " local_dofs=%" PetscInt_FMT " global_dofs=%" PetscInt_FMT " basis_functions=%" PetscInt_FMT " owned_constraints=%" PetscInt_FMT " global_constraints=%" PetscInt_FMT " analysis=%s continuation_method=%s pc_variant=%s pmg_backend=%s bc=%s omega_max=%.6g lambda_init=%.6g d_lambda_init=%.6g curve_csv=%s\n",
                        app.use_box_mesh ? "generated-box:" : "", app.use_box_mesh ? "unit" : app.mesh, app.refine_levels, basis.dim, basis.degree, cEnd - cStart, nEnd - nStart, local_dofs, global_dofs, basis.n_basis, reported_owned_constraints, reported_global_constraints, app.analysis, app.continuation_method, app.variant_name, app.pmg_apply_backend, app.mesh_bc_mode, (double)app.omega_max, (double)app.lambda_init, (double)app.d_lambda_init, app.curve_csv));
  if (app.inspect_partition) {
    PetscCall(ReportPartitionDiagnostics(dm, Areg, u, &actx, &app));
    SSR_PROFILE_TIMER_END(NULL, &run_timer, &wall_time);
    PetscCall(LinearSolverDestroy(&lsolver));
    PetscCall(AssemblyCtxDestroy(&actx));
    PetscCall(VecDestroy(&f_ext));
    PetscCall(VecDestroy(&u));
    PetscCall(MatDestroy(&Ktangent));
    PetscCall(MatDestroy(&Kelastic));
    PetscCall(MatDestroy(&Areg));
    PetscCall(DMDestroy(&dm));
    PetscCall(P4BasisDestroy(&basis));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_ASSEMBLE_ELASTIC, Kelastic, f_ext, &profile_timer);
  PetscCall(AssembleElasticProblem(&actx, Kelastic, f_ext));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elastic_assembly_time);
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_APPLY_DIRICHLET, Kelastic, f_ext, &profile_timer);
  PetscCall(ApplyZeroDirichlet(actx.constrained_is, Kelastic, f_ext));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &dirichlet_time);
  PetscCall(VecNorm(f_ext, NORM_2, &rhs_norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "SSR_ELASTIC assembly_time=%.6g dirichlet_time=%.6g rhs_norm=%.6e\n", (double)elastic_assembly_time, (double)dirichlet_time, (double)rhs_norm));
  {
    PetscBool check_symmetry = PETSC_FALSE;

    PetscCall(PetscOptionsGetBool(NULL, NULL, "-check_matrix_symmetry", &check_symmetry, NULL));
    if (check_symmetry) {
      PetscBool symmetric;

      PetscCall(MatIsSymmetric(Kelastic, 1.0e-10, &symmetric));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "elastic matrix_symmetric=%s tol=1e-10\n", symmetric ? "true" : "false"));
    }
  }
  PetscCall(VecZeroEntries(u));
  if (app.init_replay_dir[0]) {
    PetscCall(InitReplayRun(dm, &actx, &app, &lsolver, Areg, Kelastic, Ktangent, f_ext, rhs_norm));
    SSR_PROFILE_TIMER_END(NULL, &run_timer, &wall_time);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "RESULT variant=%s init_replay=true global_dofs=%" PetscInt_FMT " wall_time=%.6g deflation=%s deflation_solver=%s init_replay_dir=%s\n",
                          app.variant_name, global_dofs, (double)wall_time, app.use_deflation ? "true" : "false",
                          app.deflation_solver_name, app.init_replay_dir));
    PetscCall(LinearSolverDestroy(&lsolver));
    PetscCall(AssemblyCtxDestroy(&actx));
    PetscCall(VecDestroy(&f_ext));
    PetscCall(VecDestroy(&u));
    PetscCall(MatDestroy(&Ktangent));
    PetscCall(MatDestroy(&Kelastic));
    PetscCall(MatDestroy(&Areg));
    PetscCall(DMDestroy(&dm));
    PetscCall(P4BasisDestroy(&basis));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (app.step_replay_dir[0]) {
    NewtonStats step_stats;

    PetscCall(StepReplayRun(dm, &actx, &app, &lsolver, Areg, Kelastic, Ktangent, f_ext, rhs_norm, &step_stats));
    SSR_PROFILE_TIMER_END(NULL, &run_timer, &wall_time);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "DEFLATION_TIMING orthogonalization_time=%.6g coarse_initial_time=%.6g coarse_initial_calls=%" PetscInt_FMT " pc_apply_time=%.6g projector_time=%.6g projected_pc_calls=%" PetscInt_FMT "\n",
                          (double)lsolver.deflation_orthogonalization_time, (double)lsolver.deflation_coarse_time, lsolver.deflation_coarse_calls,
                          (double)lsolver.deflation_pc_apply_time, (double)lsolver.deflation_projector_time, lsolver.deflation_projected_pc_calls));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "RESULT variant=%s step_replay=true global_dofs=%" PetscInt_FMT " newton_its=%" PetscInt_FMT " linear_its=%" PetscInt_FMT " wall_time=%.6g deflation=%s deflation_solver=%s step_replay_dir=%s converged=%s final_rel=%.6e final_rel_correction=%.6e\n",
                          app.variant_name, global_dofs, step_stats.newton_its, step_stats.total_linear_its, (double)wall_time,
                          app.use_deflation ? "true" : "false", app.deflation_solver_name, app.step_replay_dir,
                          step_stats.converged ? "true" : "false", (double)step_stats.final_rel, (double)step_stats.final_rel_correction));
    PetscCall(LinearSolverDestroy(&lsolver));
    PetscCall(AssemblyCtxDestroy(&actx));
    PetscCall(VecDestroy(&f_ext));
    PetscCall(VecDestroy(&u));
    PetscCall(MatDestroy(&Ktangent));
    PetscCall(MatDestroy(&Kelastic));
    PetscCall(MatDestroy(&Areg));
    PetscCall(DMDestroy(&dm));
    PetscCall(P4BasisDestroy(&basis));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (app.linear_replay_dir[0]) {
    PetscCall(LinearReplayRun(dm, &actx, &app, &lsolver, Areg, Kelastic, Ktangent, f_ext, rhs_norm));
    SSR_PROFILE_TIMER_END(NULL, &run_timer, &wall_time);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "RESULT variant=%s replay=true global_dofs=%" PetscInt_FMT " wall_time=%.6g deflation=%s deflation_solver=%s replay_dir=%s\n",
                          app.variant_name, global_dofs, (double)wall_time, app.use_deflation ? "true" : "false",
                          app.deflation_solver_name, app.linear_replay_dir));
    PetscCall(LinearSolverDestroy(&lsolver));
    PetscCall(AssemblyCtxDestroy(&actx));
    PetscCall(VecDestroy(&f_ext));
    PetscCall(VecDestroy(&u));
    PetscCall(MatDestroy(&Ktangent));
    PetscCall(MatDestroy(&Kelastic));
    PetscCall(MatDestroy(&Areg));
    PetscCall(DMDestroy(&dm));
    PetscCall(P4BasisDestroy(&basis));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  {
    const SsrNativeContinuationOps *continuation_ops = NULL;

    PetscCall(SsrNativeContinuationRegistryFind(app.continuation_algorithm, &continuation_ops));
    PetscCall(continuation_ops->solve(dm, &actx, &app, &lsolver, Areg, Kelastic, Ktangent, f_ext, rhs_norm, u, &ssr_stats));
  }
  SSR_PROFILE_TIMER_END(NULL, &run_timer, &wall_time);
  {
    PetscMPIInt size;
    PetscPartitioner     part = NULL;
    PetscPartitionerType part_type = NULL;

    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCall(DMPlexGetPartitioner(dm, &part));
    if (part) PetscCall(PetscPartitionerGetType(part, &part_type));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "DEFLATION_TIMING orthogonalization_time=%.6g coarse_initial_time=%.6g coarse_initial_calls=%" PetscInt_FMT " pc_apply_time=%.6g projector_time=%.6g projected_pc_calls=%" PetscInt_FMT "\n",
                          (double)lsolver.deflation_orthogonalization_time, (double)lsolver.deflation_coarse_time, lsolver.deflation_coarse_calls,
                          (double)lsolver.deflation_pc_apply_time, (double)lsolver.deflation_projector_time, lsolver.deflation_projected_pc_calls));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "RESULT variant=%s continuation_method=%s ranks=%d partitioner=%s global_dofs=%" PetscInt_FMT " accepted_steps=%" PetscInt_FMT " total_newton_its=%" PetscInt_FMT " total_linear_its=%" PetscInt_FMT " total_line_search_its=%" PetscInt_FMT " elastic_assembly_time=%.6g continuation_wall_time=%.6g wall_time=%.6g omega_last=%.6e lambda_last=%.6e final_rel=%.6e final_rel_correction=%.6e stop_reason=%s deflation=%s deflation_solver=%s deflation_basis_cols=%" PetscInt_FMT " curve_csv=%s\n",
                          app.variant_name, app.continuation_method, size, part_type ? part_type : "unknown", global_dofs, ssr_stats.accepted_steps, ssr_stats.total_newton_its,
                          ssr_stats.total_linear_its, ssr_stats.total_line_search_its, (double)elastic_assembly_time, (double)ssr_stats.wall_time,
                          (double)wall_time, (double)ssr_stats.omega_last, (double)ssr_stats.lambda_last, (double)ssr_stats.final_rel,
                          (double)ssr_stats.final_rel_correction, ssr_stats.stop_reason, app.use_deflation ? "true" : "false", app.deflation_solver_name,
                          lsolver.n_raw_basis, app.curve_csv));
    PetscCall(WriteFinalSolutionOutputs(&app, dm, &actx, u));
    PetscCall(WriteSummaryJSON(&app, size, part_type ? part_type : "unknown", global_dofs, elastic_assembly_time, wall_time, &ssr_stats, &lsolver));
  }

  PetscCall(LinearSolverDestroy(&lsolver));
  PetscCall(AssemblyCtxDestroy(&actx));
  PetscCall(VecDestroy(&f_ext));
  PetscCall(VecDestroy(&u));
  PetscCall(MatDestroy(&Ktangent));
  PetscCall(MatDestroy(&Kelastic));
  PetscCall(MatDestroy(&Areg));
  PetscCall(DMDestroy(&dm));
  PetscCall(P4BasisDestroy(&basis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4IndirectSSRRunOptionsString(const char options[])
{
  PetscFunctionBeginUser;
  if (options && options[0]) PetscCall(PetscOptionsInsertString(NULL, options));
  PetscCall(P4IndirectSSRRunFromOptions());
  PetscFunctionReturn(PETSC_SUCCESS);
}

#ifndef P4_INDIRECT_SSR_NO_MAIN
int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, NULL, "Standalone pure PETSc P4 indirect SSR case\n"));
  PetscCall(P4IndirectSSRRunFromOptions());
  PetscCall(PetscFinalize());
  return 0;
}
#endif
