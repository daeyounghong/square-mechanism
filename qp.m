function [x, exitflag, iterations] = qp( ...
        H_size, H_row, H_col, H_dat, f, ...
        A_size, A_row, A_col, A_dat, b, ...
        Aeq_size, Aeq_row, Aeq_col, Aeq_data, beq, ...
        lb, ub, x0, ...
        optimality_tolerance, constraint_tolerance, num_cores, max_iter, step_tol)
    % $Author: Daeyoung Hong, Woohwan Jung $    $Date: 2022.02.26. $
    %{
    Args:
        The naming of arguments follows https://www.mathworks.com/help/optim/ug/quadprog.html.
        H_size: the size of H for function `quadprog` in MATLAB (e.g., for 5 X 3 matrix, H_size = [5 3])
        H_row: row indices of non-zero elements in H for function `quadprog`
        H_col: column indices of non-zero elements in H for function `quadprog`
        H_dat: the values of non-zero elements in H for function `quadprog`
        f: f for function `quadprog`
        A_row: row indices of non-zero elements in A for function `quadprog`
        A_col: column indices of non-zero elements in A for function `quadprog`
        A_dat: the values of non-zero elements in A for function `quadprog`
        b: b for function `quadprog`
        Aeq_row: row indices of non-zero elements in Aeq for function `quadprog`
        Aeq_col: column indices of non-zero elements in Aeq for function `quadprog`
        Aeq_dat: the values of non-zero elements in Aeq for function `quadprog`
        beq, lb, ub, x0: See https://www.mathworks.com/help/optim/ug/quadprog.html.
        optimality_tolerance: `OptimalityTolerance` for function `quadprog`
        constraint_tolerance: `ConstraintTolerance` for function `quadprog`
        num_cores: the number of threads for multithreading
        max_iter: `MaxIterations` for function `quadprog`
        step_tol: `StepTolerance` for function `quadprog`
    %}
    disp('num_cores = ');
    disp(num_cores);
    if num_cores >= 2
        prev_num_cores = maxNumCompThreads(num_cores)
    end
    H = sparse(H_row, H_col, H_dat, H_size(1), H_size(2));
    if all(size(A_dat) == [0 0])
        A = []
    else
        A = sparse(A_row, A_col, A_dat, A_size(1), A_size(2));
    end
    disp(size(A));
    disp(size(b));
    Aeq = sparse(Aeq_row, Aeq_col, Aeq_data, Aeq_size(1), Aeq_size(2));
    clear H_size H_row H_col H_dat;
    clear A_size A_row A_col A_dat;
    clear Aeq_size Aeq_row Aeq_col Aeq_data;
    options = optimoptions( ...
        'quadprog','Display','iter', ...
        'MaxIterations', max_iter, ...
        'OptimalityTolerance', optimality_tolerance, ...
        'ConstraintTolerance', constraint_tolerance, ...
        'StepTolerance', step_tol)
    disp('MATLAB: start to optimize');
    [x, fval, exitflag, output] = quadprog(H, f, A, b, Aeq, beq, lb, ub, x0, options);
    iterations = output.iterations