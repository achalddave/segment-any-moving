function evaluate(groundtruth_dir, predictions_mat, eval_code_path)
    predictions = load(predictions_mat);
    predictions = predictions.predictions;

    groundtruth_pngs = dir([groundtruth_dir '/*.png']);
    groundtruth_pngs = sort({groundtruth_pngs.name});
    num_groundtruth = size(groundtruth_pngs, 2);
    assert(num_groundtruth == size(predictions, 1));
    groundtruth = zeros(size(predictions));
    for i = 1:num_groundtruth
        groundtruth(i, :, :) = imread(...
            [groundtruth_dir '/' groundtruth_pngs{i}]);
    end

    addpath(genpath(eval_code_path));
    [F, P, R, obj_detected, obj_detected_075, obj_gt] = accuracy(groundtruth, predictions);
    fprintf('F: %.2f\n', 100*F)
    fprintf('P: %.2f\n', 100*P)
    fprintf('R: %.2f\n', 100*R)
    fprintf('N0.75: %d\n', obj_detected_075)
    fprintf('N-predicted: %d\n', obj_detected)
    fprintf('N-groundtruth: %d\n', obj_gt)
end
