function train_logistic_model(cidx1,cidx2,X_pos,X_neg,pairwiseDir)

fprintf('train_spatial_diff_part_app_dx_dy()\n');
% training flags
is_neighbor_locref = true;

if (ischar(cidx1))
    cidx1 = str2num(cidx1);
end

if (ischar(cidx2))
    cidx2 = str2num(cidx2);
end

if (~exist(pairwiseDir,'dir')),mkdir(pairwiseDir);end

modelFname = [pairwiseDir '/spatial_model_cidx_' num2str(cidx1) '_' num2str(cidx2)];
try
    %assert(false);
    load(modelFname,'spatial_model');
    fprintf('cidx: %d - %d\n',[cidx1,cidx2]);
    fprintf('spatial model file loaded. quitting.\n');
catch
    opts.X_pos_mean = mean(X_pos);
    opts.X_pos_std = std(X_pos);
    %idxs1 = X_pos(:,1) >= opts.X_pos_mean(:,1) - 3*opts.X_pos_std(:,1) & X_pos(:,1) <= opts.X_pos_mean(:,1) + 3*opts.X_pos_std(:,1);
    %idxs2 = X_pos(:,2) >= opts.X_pos_mean(:,2) - 3*opts.X_pos_std(:,2) & X_pos(:,2) <= opts.X_pos_mean(:,2) + 3*opts.X_pos_std(:,2);
    %idxs3 = X_pos(:,3) >= opts.X_pos_mean(:,3) - 3*opts.X_pos_std(:,3) & X_pos(:,3) <= opts.X_pos_mean(:,3) + 3*opts.X_pos_std(:,3);
    %X_pos = X_pos(idxs1 & idxs2 & idxs3,:);
    if is_neighbor_locref
        X_pos = get_augm_spatial_features_diff_neighbour_locref(X_pos); %opts.X_pos_mean, p
        X_neg = get_augm_spatial_features_diff_neighbour_locref(X_neg); %opts.X_pos_mean, p
    else
        X_pos = get_augm_spatial_features_diff_neighbour(X_pos, p); %opts.X_pos_mean, p
        X_neg = get_augm_spatial_features_diff_neighbour(X_neg, p); %opts.X_pos_mean, p
    end

    if (size(X_neg,1) > size(X_pos,1))
        nFeat = size(X_pos,1);
        idxs_rnd = randperm(size(X_neg,1));
        idxsSamp = idxs_rnd(1:nFeat);
        X_neg = X_neg(idxsSamp,:);
    end

    [X_norm, opts.X_min, opts.X_max] = getFeatNorm([X_pos;X_neg]);
    X_pos_norm = X_norm(1:size(X_pos,1),:);
    X_neg_norm = X_norm(size(X_pos,1)+1:end,:);
    %X_pos_norm = X_pos;
    %X_neg_norm = X_neg;

    reg_type = 0; % L2
    C = 1e-3;

    nPos = size(X_pos_norm,1);
    nNeg = size(X_neg_norm,1);
    lab_pos = ones(nPos,1);
    lab_neg = zeros(nNeg,1);
    lab = [lab_pos; lab_neg];
    ex = sparse(double([X_pos_norm; X_neg_norm]));
    model = train(lab, ex, ['-s ' num2str(reg_type) ' -B 1 -c ' num2str(C)]);

    spatial_model.training_opts = opts;
    spatial_model.log_reg = model;
    save(modelFname, 'spatial_model');

    if (0)
        visDir = [pairwiseDir '/vis/'];
        if (~exist(visDir,'dir')),mkdir(visDir);end
        [~,acc,pred] = predict(lab, ex, model, '-b 1');
        scrsz = get(0,'ScreenSize');
        figure('Position',[1 scrsz(4) scrsz(3)/2 scrsz(4)]);
        vis_logreg(pred,acc,1:nPos,1+nPos:nPos+nNeg);
        print(gcf,'-dpng',[visDir '/logreg_cidx_' num2str(cidx1) '_' num2str(cidx2) '.png']);
        close all;
    end
end
% ------------------------------------------------------------------------
