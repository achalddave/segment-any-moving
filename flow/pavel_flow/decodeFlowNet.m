function decodeFlowNet(floDir, outputDir, overwrite)
    if nargin < 3
        overwrite = false;
    end
    flows = dir([floDir, '/*.flo']);
    for j = 1:length(flows)
        frameName = strsplit(flows(j).name, '.');
        outputAngle = [outputDir, '/angleField_', frameName{1}, '.jpg'];
        outputMagnitude = [outputDir, '/magField_', frameName{1}, '.jpg'];
        outputMinMax = [outputDir, '/minmax.txt'];
        if ~overwrite && exist(outputAngle, 'file') && ...
                         exist(outputMagnitude, 'file') && ...
                         exist(outputMinMax, 'file')
            continue
        end

        flow = readFloFile([floDir, '/', flows(j).name]);

        baseVector = zeros(size(flow, 1), size(flow, 2), 2);
        baseVector(:, :, 1) = 1;
        angleField = acos(dot(flow, baseVector, 3) ./ (sqrt(flow(:, :, 1).^2 + flow(:, :, 2).^2) .* ones(size(flow, 1), size(flow, 2))));
        magnitudes = sqrt(flow(:, :, 1).^2 + flow(:, :, 2).^2);

        min_angle = min(angleField(:));
        max_angle = max(angleField(:));

        angleField = (angleField - min_angle) ./ (max_angle - min_angle);

        imwrite(angleField, outputAngle);

        min_magnitude = min(magnitudes(:));
        max_magnitude = max(magnitudes(:));

        magnitudes = (magnitudes - min_magnitude) ./ (max_magnitude - min_magnitude);

        imwrite(magnitudes, outputMagnitude);
        fid = fopen(outputMinMax, 'a');
        fprintf(fid, '%f %f %f %f\n', min_angle, max_angle, min_magnitude, max_magnitude);
        fclose(fid);
    end
end
