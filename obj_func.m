function [loss, real, pred] = obj_func(u, data, thresh, opts)
    % General parameters for objective function
    data_length = size(data, 1);
    data = data(1:data_length, :);
    w = u(1:end-1);
    b = u(end);
    class = data(:, 1);
    labels = data(:, 2:end);
    preds = zeros(data_length, 1);
    
    Pred = 0;
    
    TP = 0;
    FP = 0;
    
    TN = 0;
    FN = 0;
    
    for i=1:data_length
    
        if opts == "linear"
            obj_val = 1/(1 + exp(-w*labels(i, :)' + b));
            if obj_val >= thresh 
                preds(i) = 0.9;
                Pred = 1;
            elseif obj_val <= thresh
                preds(i) = 0.1;
                Pred = 0;
            end
        end
        
        if (class(i) == 1) && (Pred == 1)
            TP = TP + 1;
        elseif (class(i) == 1) && (Pred == 0)
            FP = FP + 1;
        end
        
        if (class(i) == 0) && (Pred == 0)
            TN = TN + 1;
        elseif (class(i) == 0) && (Pred == 1)
            FN = FN + 1;
        end
        
        
        
        
        
    end
    
    sensitivity = TP/(TP + FN + eps);
    specificity = TN/(TN + FP + eps);
    
    accuracy = (sensitivity + specificity)/2;
        
    if opts == "linear"
        loss = 5/accuracy...
            -1*sum(class.*log(preds) + (1 - class).*log(1 - preds))/length(class);      
        
    end
    real = class;
    pred = preds;

end