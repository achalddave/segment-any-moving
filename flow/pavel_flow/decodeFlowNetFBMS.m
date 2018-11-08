function decodeFlowNetFBMS(set)    
    vids = dir(['/scratch/clear/ptokmako/datasets/FBMS/', set, '/Frames']);
    for i = 3 : length(vids)
        mkdir(['/scratch/clear/ptokmako/datasets/FBMS/', set, '/FlowNetFlowAngle/', vids(i).name]);
        flows = dir(['/scratch/clear/ptokmako/datasets/FBMS/', set, '/FlowNetFlow/', vids(i).name, '/*.flo']);
        for j = 1 : length(flows)
            flow = readFloFile(['/scratch/clear/ptokmako/datasets/FBMS/', set, '/FlowNetFlow/', vids(i).name, '/', flows(j).name]);        

            baseVector = zeros(size(flow, 1), size(flow, 2), 2);
            baseVector(:, :, 1) = 1;
            angleField = acos(dot(flow, baseVector, 3) ./ (sqrt(flow(:, :, 1).^2 + flow(:, :, 2).^2) .* ones(size(flow, 1), size(flow, 2))));
            magnitudes = sqrt(flow(:, :, 1).^2 + flow(:, :, 2).^2);

            min_angle = min(angleField(:));
            max_angle = max(angleField(:));

            angleField = (angleField - min_angle) ./ (max_angle - min_angle);
            
            frameName = strsplit(flows(j).name, '.');
            imwrite(angleField, ['/scratch/clear/ptokmako/datasets/FBMS/', set, '/FlowNetFlowAngle/', vids(i).name, '/angleField_', frameName{1}, '.jpg']);

            min_magnitude = min(magnitudes(:));
            max_magnitude = max(magnitudes(:));

            magnitudes = (magnitudes - min_magnitude) ./ (max_magnitude - min_magnitude);

            imwrite(magnitudes, ['/scratch/clear/ptokmako/datasets/FBMS/', set, '/FlowNetFlowAngle/', vids(i).name, '/magField_', frameName{1}, '.jpg']);
            fid = fopen(['/scratch/clear/ptokmako/datasets/FBMS/', set, '/FlowNetFlowAngle/', vids(i).name, '/minmax.txt'], 'a');
            fprintf(fid, '%f %f %f %f\n', min_angle, max_angle, min_magnitude, max_magnitude);
            fclose(fid); 
        end
    end
   
end
