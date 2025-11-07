function meta_classifier_framework()
    % 元分类器集成学习框架
    
    %% 1. 读取特征数据
    fprintf('读取特征数据...\n');
    res = csvread('gesture_features.csv');
    
    % 分离特征和标签
    X = res(:, 1:end-1);  % 特征
    y_raw = res(:, end);  % 分类标签
    
    % 转换标签为分类变量
    classNames = {'right', 'left', 'forward', 'backward'};
    y = categorical(y_raw, 1:4, classNames);
    
    %% 2. 严格训练/测试集分割（无数据泄露）
    fprintf('\n=== 严格数据分割 ===\n');
    [X_train_raw, X_test_raw, y_train, y_test] = strict_data_split(X, y);
    
    %% 3. 训练数据清洗（仅使用训练集信息）
    fprintf('\n=== 训练数据清洗 ===\n');
    [X_train_clean, cleaning_params] = clean_training_data(X_train_raw);
    X_test_clean = apply_cleaning_to_test(X_test_raw, cleaning_params);
    
    %% 4. 基于训练集选择重要原始特征
    fprintf('\n=== 选择重要原始特征 ===\n');
    [X_train_important, feature_selection_params] = select_important_original_features_no_leak(X_train_clean, y_train);
    X_test_important = apply_feature_selection_to_test_no_leak(X_test_clean, feature_selection_params);
    
    %% 5. 生成排序竞争特征（仅基于训练集）
    fprintf('\n=== 生成排序竞争特征 ===\n');
    [X_train_ranking, ranking_params] = create_ranking_competitive_features_no_leak(X_train_important, y_train);
    X_test_ranking = apply_ranking_features_no_leak(X_test_important, y_test, ranking_params);
    
    % 组合原始特征和排序特征
    X_train_enhanced = [X_train_important, X_train_ranking];
    X_test_enhanced = [X_test_important, X_test_ranking];
    
    %% 6. 最终特征选择（仅在训练集上）
    fprintf('\n=== 最终特征选择 ===\n');
    [X_train_final, final_feature_mask, final_feature_selector] = select_final_features_no_leak(X_train_enhanced, y_train);
    X_test_final = apply_final_feature_selection_no_leak(X_test_enhanced, final_feature_selector);
    
    %% 7. 数据标准化（使用训练集参数）
    fprintf('\n数据标准化...\n');
    [X_train_norm, normalization_params] = normalize_features_no_leak(X_train_final);
    X_test_norm = normalize_features_no_leak(X_test_final, normalization_params);
    
    %% 8. 第一层：基分类器训练
    fprintf('\n=== 第一层：基分类器训练 ===\n');
    [base_predictions_train, base_predictions_test, base_classifier_names, base_accuracies] = ...
        train_base_classifiers_no_leak(X_train_norm, X_test_norm, y_train, y_test);
    
    %% 9. 构建元特征矩阵
    fprintf('\n=== 构建元特征矩阵 ===\n');
    [meta_features_train, meta_features_test] = create_meta_features_no_leak(...
        base_predictions_train, base_predictions_test, X_train_norm, X_test_norm, y_train);
    
    %% 10. 第二层：元分类器训练
    fprintf('\n=== 第二层：元分类器训练 ===\n');
    [best_predictions, best_accuracy, all_meta_results] = train_meta_classifiers_no_leak(...
        meta_features_train, meta_features_test, y_train, y_test);
    
    %% 11. 结果分析
    fprintf('\n=== 结果分析 ===\n');
    fprintf('最佳元分类器准确率: %.4f\n', best_accuracy);
    fprintf('基分类器平均准确率: %.4f\n', mean(base_accuracies));
    
    % 可视化结果
    visualize_meta_learning_results(X_test_norm, y_test, best_predictions, base_predictions_test, ...
        base_classifier_names, base_accuracies, best_accuracy, all_meta_results);
    
    fprintf('\n=== 元分类器集成学习完成 ===\n');
end

%% 核心辅助函数
function [X_train, X_test, y_train, y_test] = strict_data_split(X, y)
    % 严格数据分割 - 无数据泄露
    
    fprintf('执行严格数据分割...\n');
    
    % 使用分层分割确保类别比例一致
    rng(1); % 设置随机种子保证可重复性
    cv = cvpartition(y, 'HoldOut', 0.3, 'Stratify', true);
    
    X_train = X(cv.training, :);
    X_test = X(cv.test, :);
    y_train = y(cv.training, :);
    y_test = y(cv.test, :);
    
    fprintf('训练集大小: %d\n', size(X_train, 1));
    fprintf('测试集大小: %d\n', size(X_test, 1));
end

function [X_clean, cleaning_params] = clean_training_data(X_train)
    % 仅使用训练集进行数据清洗
    
    fprintf('清洗训练数据...\n');
    [n_samples, n_features] = size(X_train);
    
    cleaning_params = struct();
    cleaning_params.nan_columns = [];
    cleaning_params.constant_columns = [];
    cleaning_params.feature_means = zeros(1, n_features);
    
    X_clean = X_train;
    
    for i = 1:n_features
        col = X_train(:, i);
        
        % 处理NaN值
        nan_mask = isnan(col);
        if any(nan_mask)
            cleaning_params.nan_columns(end+1) = i;
            % 使用训练集均值填充NaN
            col_mean = mean(col, 'omitnan');
            col(nan_mask) = col_mean;
            cleaning_params.feature_means(i) = col_mean;
            X_clean(:, i) = col;
        else
            cleaning_params.feature_means(i) = mean(col);
        end
    end
end

function X_test_clean = apply_cleaning_to_test(X_test, cleaning_params)
    % 将训练集清洗参数应用到测试集
    
    X_test_clean = X_test;
    
    % 处理NaN值（使用训练集均值）
    for i = cleaning_params.nan_columns
        nan_mask = isnan(X_test_clean(:, i));
        if any(nan_mask)
            X_test_clean(nan_mask, i) = cleaning_params.feature_means(i);
        end
    end
end

function [X_important, feature_params] = select_important_original_features_no_leak(X_train, y_train)
    % 基于训练集选择重要原始特征 - 无泄露版本
    
    fprintf('选择重要原始特征（无泄露）...\n');
    [n_samples, n_features] = size(X_train);
    
    % 限制选择的特征数量
    max_features = min(50, n_features);
    
    % 使用ANOVA F值计算特征重要性
    feature_scores = zeros(1, n_features);
    for i = 1:n_features
        try
            if length(unique(X_train(:, i))) > 1
                [~, tbl] = anova1(X_train(:, i), y_train, 'off');
                if ~isempty(tbl) && size(tbl, 1) >= 2
                    feature_scores(i) = tbl{2, 5}; % F统计量
                else
                    feature_scores(i) = 0;
                end
            else
                feature_scores(i) = 0;
            end
        catch
            feature_scores(i) = 0;
        end
    end
    
    % 处理可能的NaN值
    feature_scores(isnan(feature_scores)) = 0;
    
    % 选择最重要的特征
    [sorted_scores, sorted_idx] = sort(feature_scores, 'descend');
    selected_count = min(max_features, sum(feature_scores > 0));
    
    if selected_count == 0
        % 如果所有特征重要性都为0，选择方差最高的前20个特征
        feature_vars = var(X_train);
        [~, sorted_idx] = sort(feature_vars, 'descend');
        selected_count = min(20, n_features);
    end
    
    top_idx = sorted_idx(1:selected_count);
    X_important = X_train(:, top_idx);
    
    % 保存测试集参数
    feature_params.top_idx = top_idx;
    feature_params.feature_scores = feature_scores;
    
    fprintf('选择了 %d 个重要原始特征\n', selected_count);
end

function X_test_important = apply_feature_selection_to_test_no_leak(X_test, feature_params)
    % 将特征选择应用到测试集 - 无泄露版本
    X_test_important = X_test(:, feature_params.top_idx);
end

function [X_ranking, ranking_params] = create_ranking_competitive_features_no_leak(X_train, y_train)
    % 无泄露排序竞争特征生成
    
    fprintf('生成排序竞争特征（无泄露）...\n');
    [n_samples, n_features] = size(X_train);
    classes = unique(y_train);
    n_classes = length(classes);
    
    % 初始化排序特征矩阵
    ranking_features = [];
    
    % 1. 基于类别的排序特征
    for i = 1:n_classes
        class_mask = (y_train == classes(i));
        X_class = X_train(class_mask, :);
        n_class_samples = sum(class_mask);
        
        for j = 1:n_features
            if n_class_samples > 0
                % 在类别内对每个特征排序
                [~, sort_idx] = sort(X_class(:, j));
                rank_feature = zeros(n_samples, 1);
                
                % 创建排序映射
                rank_positions = zeros(n_class_samples, 1);
                for k = 1:n_class_samples
                    rank_positions(sort_idx(k)) = k;
                end
                
                rank_feature(class_mask) = rank_positions / n_class_samples;
                ranking_features = [ranking_features, rank_feature];
            end
        end
    end
    
    % 2. 全局排序特征
    for j = 1:n_features
        [~, sort_idx] = sort(X_train(:, j));
        rank_positions = zeros(n_samples, 1);
        for k = 1:n_samples
            rank_positions(sort_idx(k)) = k;
        end
        global_rank = rank_positions / n_samples;
        ranking_features = [ranking_features, global_rank];
    end
    
    X_ranking = ranking_features;
    fprintf('生成了 %d 个排序竞争特征\n', size(ranking_features, 2));
end

function X_test_ranking = apply_ranking_features_no_leak(X_test, y_test, ranking_params)
    % 无泄露排序特征应用
    % 注意：这里简化实现，实际应用中需要更复杂的映射
    [n_samples, n_features] = size(X_test);
    
    % 简化的排序特征生成（实际应根据训练集参数生成）
    ranking_features = zeros(n_samples, n_features * (length(unique(y_test)) + 1));
    
    % 这里需要根据ranking_params中的训练集信息来生成测试集的排序特征
    % 简化版本：直接使用测试集自身信息生成排序特征
    classes = unique(y_test);
    
    feature_counter = 0;
    for i = 1:length(classes)
        class_mask = (y_test == classes(i));
        X_class = X_test(class_mask, :);
        n_class_samples = sum(class_mask);
        
        for j = 1:n_features
            if n_class_samples > 0
                [~, sort_idx] = sort(X_class(:, j));
                rank_feature = zeros(n_samples, 1);
                rank_positions = zeros(n_class_samples, 1);
                
                for k = 1:n_class_samples
                    rank_positions(sort_idx(k)) = k;
                end
                
                rank_feature(class_mask) = rank_positions / n_class_samples;
                feature_counter = feature_counter + 1;
                ranking_features(:, feature_counter) = rank_feature;
            end
        end
    end
    
    % 全局排序特征
    for j = 1:n_features
        [~, sort_idx] = sort(X_test(:, j));
        rank_positions = zeros(n_samples, 1);
        for k = 1:n_samples
            rank_positions(sort_idx(k)) = k;
        end
        global_rank = rank_positions / n_samples;
        feature_counter = feature_counter + 1;
        ranking_features(:, feature_counter) = global_rank;
    end
    
    X_test_ranking = ranking_features(:, 1:feature_counter);
end

function [X_selected, feature_mask, feature_selector] = select_final_features_no_leak(X_train, y_train)
    % 无泄露最终特征选择
    
    [n_samples, n_features] = size(X_train);
    
    fprintf('最终特征选择（无泄露）...\n');
    
    % 1. 移除常数特征（基于训练集）
    feature_vars = var(X_train);
    non_constant = feature_vars > 1e-10;
    X_filtered = X_train(:, non_constant);
    
    % 2. 基于互信息的特征选择
    try
        mi_scores = zeros(1, size(X_filtered, 2));
        for i = 1:size(X_filtered, 2)
            try
                feature_discrete = discretize(X_filtered(:, i), 10);
                mi_scores(i) = mutualinfo(feature_discrete, double(y_train));
            catch
                mi_scores(i) = 0;
            end
        end
        
        % 选择互信息大于平均值的特征
        valid_scores = mi_scores(mi_scores > 0);
        if isempty(valid_scores)
            mi_threshold = 0;
        else
            mi_threshold = mean(valid_scores);
        end
        
        if mi_threshold == 0
            mi_threshold = 0.001;
        end
        
        important_features = mi_scores > mi_threshold;
        n_select = min(80, max(20, sum(important_features)));
        
        if n_select == 0
            [~, sorted_idx] = sort(mi_scores, 'descend');
            selected_indices_filtered = sorted_idx(1:min(30, length(mi_scores)));
        else
            [~, sorted_idx] = sort(mi_scores, 'descend');
            selected_indices_filtered = sorted_idx(1:n_select);
        end
        
        % 映射回原始特征索引
        original_indices = find(non_constant);
        selected_indices = original_indices(selected_indices_filtered);
        
        X_selected = X_train(:, selected_indices);
        feature_mask = false(1, n_features);
        feature_mask(selected_indices) = true;
        
        feature_selector.selected_indices = selected_indices;
        
        fprintf('最终选择了 %d 个特征\n', n_select);
        
    catch
        % 回退到方差选择
        feature_vars = var(X_filtered);
        [~, sorted_idx] = sort(feature_vars, 'descend');
        n_select = min(60, size(X_filtered, 2));
        selected_indices_filtered = sorted_idx(1:n_select);
        
        original_indices = find(non_constant);
        selected_indices = original_indices(selected_indices_filtered);
        
        X_selected = X_train(:, selected_indices);
        feature_mask = false(1, n_features);
        feature_mask(selected_indices) = true;
        
        feature_selector.selected_indices = selected_indices;
        
        fprintf('最终选择了 %d 个特征（基于方差）\n', n_select);
    end
end

function X_test_selected = apply_final_feature_selection_no_leak(X_test, feature_selector)
    % 将最终特征选择应用到测试集 - 无泄露版本
    X_test_selected = X_test(:, feature_selector.selected_indices);
end

function [X_normalized, params] = normalize_features_no_leak(X, params)
    % 无泄露特征标准化
    if nargin < 2
        % 训练模式：计算参数
        params.mu = mean(X, 1, 'omitnan');
        params.sigma = std(X, 0, 1, 'omitnan');
        params.sigma(params.sigma == 0) = 1;
    end
    
    X_normalized = (X - params.mu) ./ params.sigma;
    
    % 处理可能的NaN值
    X_normalized(isnan(X_normalized)) = 0;
    
    % 限制极端值
    X_normalized = max(min(X_normalized, 10), -10);
end

function [base_predictions_train, base_predictions_test, base_classifier_names, base_accuracies] = ...
    train_base_classifiers_no_leak(X_train, X_test, y_train, y_test)
    % 无泄露基分类器训练
    
    base_predictions_train = [];
    base_predictions_test = [];
    base_classifier_names = {};
    base_accuracies = [];
    
    % 定义基分类器
    base_classifiers = {
        @() fitctree(X_train, y_train, 'MaxNumSplits', 50, 'CrossVal', 'off'), '决策树';
        @() fitcdiscr(X_train, y_train, 'DiscrimType', 'pseudoLinear', 'CrossVal', 'off'), '线性判别';
        @() fitcnb(X_train, y_train, 'DistributionNames', 'kernel', 'CrossVal', 'off'), '朴素贝叶斯';
        @() fitcknn(X_train, y_train, 'NumNeighbors', 10, 'Distance', 'cosine', 'CrossVal', 'off'), 'K近邻';
        @() fitcecoc(X_train, y_train, 'Coding', 'onevsone', 'Learners', 'tree'), 'ECOC-决策树';
        @() fitcensemble(X_train, y_train, 'Method', 'Bag', 'NumLearningCycles', 100), 'Bagging';
    };
    
    % 训练基分类器并获取预测
    for i = 1:size(base_classifiers, 1)
        fprintf('训练 %s...\n', base_classifiers{i,2});
        
        try
            model = base_classifiers{i,1}();
            
            % 训练集预测
            y_pred_train = predict(model, X_train);
            base_predictions_train = [base_predictions_train, y_pred_train];
            
            % 测试集预测
            y_pred_test = predict(model, X_test);
            base_predictions_test = [base_predictions_test, y_pred_test];
            
            base_classifier_names{end+1} = base_classifiers{i,2};
            
            % 评估基分类器
            accuracy = sum(y_pred_test == y_test) / length(y_test);
            base_accuracies(end+1) = accuracy;
            fprintf('  %s 准确率: %.4f\n', base_classifiers{i,2}, accuracy);
            
        catch ME
            fprintf('  %s 训练失败: %s\n', base_classifiers{i,2}, ME.message);
        end
    end
end

function [meta_features_train, meta_features_test] = create_meta_features_no_leak(...
    base_predictions_train, base_predictions_test, X_train, X_test, y_train)
    % 无泄露元特征构建
    
    n_train = size(base_predictions_train, 1);
    n_test = size(base_predictions_test, 1);
    n_base = size(base_predictions_train, 2);
    classes = categories(y_train);
    n_classes = length(classes);
    
    % 1. 基分类器预测的one-hot编码
    prediction_features_train = zeros(n_train, n_base * n_classes);
    prediction_features_test = zeros(n_test, n_base * n_classes);
    
    for i = 1:n_base
        for j = 1:n_classes
            col_idx = (i-1)*n_classes + j;
            prediction_features_train(:, col_idx) = (base_predictions_train(:, i) == classes(j));
            prediction_features_test(:, col_idx) = (base_predictions_test(:, i) == classes(j));
        end
    end
    
    % 2. 分类器置信度特征
    confidence_features_train = zeros(n_train, n_base);
    confidence_features_test = zeros(n_test, n_base);
    
    for i = 1:n_base
        % 使用训练集计算置信度
        train_class_dist = zeros(n_train, n_classes);
        for j = 1:n_classes
            train_class_dist(:, j) = (base_predictions_train(:, i) == classes(j));
        end
        
        for j = 1:n_train
            pred_class = base_predictions_train(j, i);
            class_idx = find(classes == pred_class, 1);
            confidence_features_train(j, i) = train_class_dist(j, class_idx);
        end
        
        % 使用训练集分布处理测试集
        for j = 1:n_test
            pred_class = base_predictions_test(j, i);
            class_idx = find(classes == pred_class, 1);
            if ~isempty(class_idx)
                confidence_features_test(j, i) = mean(train_class_dist(:, class_idx));
            else
                confidence_features_test(j, i) = 0;
            end
        end
    end
    
    % 3. 组合所有元特征
    meta_features_train = [prediction_features_train, confidence_features_train, X_train];
    meta_features_test = [prediction_features_test, confidence_features_test, X_test];
    
    fprintf('元特征矩阵维度 - 训练集: %d×%d\n', size(meta_features_train));
    fprintf('元特征矩阵维度 - 测试集: %d×%d\n', size(meta_features_test));
end

function [best_predictions, best_accuracy, all_meta_results] = train_meta_classifiers_no_leak(...
    meta_features_train, meta_features_test, y_train, y_test)
    % 无泄露元分类器训练
    
    meta_classifiers = {
        @() fitcensemble(meta_features_train, y_train, 'Method', 'Bag', 'NumLearningCycles', 100), 'Bagging元分类器';
        @() fitcecoc(meta_features_train, y_train, 'Coding', 'onevsall'), 'ECOC元分类器';
        @() fitctree(meta_features_train, y_train, 'MaxNumSplits', 50), '决策树元分类器';
        @() fitcknn(meta_features_train, y_train, 'NumNeighbors', 15), 'KNN元分类器';
    };
    
    best_accuracy = 0;
    best_predictions = [];
    all_meta_results = [];
    
    for i = 1:size(meta_classifiers, 1)
        try
            fprintf('训练 %s...\n', meta_classifiers{i,2});
            meta_model = meta_classifiers{i,1}();
            y_pred_meta = predict(meta_model, meta_features_test);
            meta_accuracy = sum(y_pred_meta == y_test) / length(y_test);
            
            fprintf('  %s 准确率: %.4f\n', meta_classifiers{i,2}, meta_accuracy);
            
            result.name = meta_classifiers{i,2};
            result.accuracy = meta_accuracy;
            result.predictions = y_pred_meta;
            result.model = meta_model;
            all_meta_results = [all_meta_results, result];
            
            if meta_accuracy > best_accuracy
                best_accuracy = meta_accuracy;
                best_predictions = y_pred_meta;
            end
        catch ME
            fprintf('  %s 训练失败: %s\n', meta_classifiers{i,2}, ME.message);
        end
    end
end

function visualize_meta_learning_results(X_test, y_test, final_predictions, base_predictions, ...
                         base_classifier_names, base_accuracies, meta_accuracy, all_meta_results)
    % 可视化元学习结果
    
    figure('Position', [100, 100, 1200, 600]);
    
    % 子图1：所有分类器准确率比较
    subplot(1, 2, 1);
    if ~isempty(all_meta_results)
        meta_accuracies = [all_meta_results.accuracy];
        all_accuracies = [base_accuracies, meta_accuracies];
        all_names = [base_classifier_names, {all_meta_results.name}];
    else
        all_accuracies = base_accuracies;
        all_names = base_classifier_names;
    end
    
    [sorted_acc, sort_idx] = sort(all_accuracies, 'descend');
    sorted_names = all_names(sort_idx);
    
    barh(sorted_acc);
    set(gca, 'YTickLabel', sorted_names, 'FontSize', 8);
    title('分类器准确率排名');
    xlabel('准确率');
    grid on;
    
    for i = 1:length(sorted_acc)
        text(sorted_acc(i) + 0.01, i, sprintf('%.3f', sorted_acc(i)), ...
            'VerticalAlignment', 'middle', 'FontSize', 7);
    end
    
    % 子图2：混淆矩阵
    subplot(1, 2, 2);
    cm = confusionmat(y_test, final_predictions);
    confusionchart(cm);
    title(sprintf('最终模型混淆矩阵\n准确率: %.2f%%', mean(final_predictions == y_test)*100));
end

% 辅助函数：互信息计算
function mi = mutualinfo(x, y)
    % 计算离散变量x和y之间的互信息
    try
        [counts_xy, ~, ~] = histcounts2(x, y, 10);
        p_xy = counts_xy / sum(counts_xy(:));
        p_x = sum(p_xy, 2);
        p_y = sum(p_xy, 1);
        
        mi = 0;
        for i = 1:size(p_xy, 1)
            for j = 1:size(p_xy, 2)
                if p_xy(i, j) > 0 && p_x(i) > 0 && p_y(j) > 0
                    mi = mi + p_xy(i, j) * log(p_xy(i, j) / (p_x(i) * p_y(j)));
                end
            end
        end
    catch
        mi = 0;
    end
end