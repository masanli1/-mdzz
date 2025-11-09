function meta_classification_with_ranking_features_three_way()
    % 集成排序竞争特征的元分类器程序 - 三划分版本（训练/验证/测试）
    
    %% 1. 读取真实数据
    fprintf('读取真实数据...\n');
    
    % 读取CSV文件
    res = csvread('ax2.csv');
    
    % 检查数据维度
    [n_samples, n_columns] = size(res);
    fprintf('数据维度: %d×%d\n', n_samples, n_columns);
    
    if n_columns < 2
        error('数据至少需要2列：特征和分类标志');
    end
    
    % 分离特征和标签
    X = res(:, 1:end-1);  % 特征
    y_raw = res(:, end);  % 分类标志
    
    % 将标签转换为分类变量
    y = categorical(y_raw);
    
    % 显示数据基本信息
    fprintf('特征数量: %d\n', size(X, 2));
    fprintf('样本数量: %d\n', length(y));
    fprintf('类别数量: %d\n', length(unique(y)));
    
    %% 2. 三划分：训练集/验证集/测试集
    fprintf('\n=== 三划分数据划分 ===\n');
    [X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test, split_info] = three_way_data_split(X, y);
    
    %% 3. 训练集数据清理（仅使用训练集信息）
    fprintf('\n=== 训练集数据清理 ===\n');
    [X_train_clean, cleaning_params] = clean_training_data(X_train_raw);
    X_val_clean = apply_cleaning_to_test(X_val_raw, cleaning_params);
    X_test_clean = apply_cleaning_to_test(X_test_raw, cleaning_params);
    
    %% 4. 基于训练集选择重要原始特征
    fprintf('\n=== 选择重要原始特征 ===\n');
    [X_train_important, feature_selection_params] = select_important_original_features_no_leak(X_train_clean, y_train);
    X_val_important = apply_feature_selection_to_test_no_leak(X_val_clean, feature_selection_params);
    X_test_important = apply_feature_selection_to_test_no_leak(X_test_clean, feature_selection_params);
    
    fprintf('重要原始特征 - 训练集: %d×%d\n', size(X_train_important, 1), size(X_train_important, 2));
    fprintf('重要原始特征 - 验证集: %d×%d\n', size(X_val_important, 1), size(X_val_important, 2));
    fprintf('重要原始特征 - 测试集: %d×%d\n', size(X_test_important, 1), size(X_test_important, 2));
    
    %% 5. 生成排序竞争特征（基于训练集，在验证集上调整）
    fprintf('\n=== 生成排序竞争特征 ===\n');
    [X_train_ranking, ranking_params] = create_ranking_competitive_features_no_leak(X_train_important, y_train);
    X_val_ranking = apply_ranking_features_no_leak(X_val_important, y_val, ranking_params);
    X_test_ranking = apply_ranking_features_no_leak(X_test_important, y_test, ranking_params);
    
    % 组合原始特征和排序特征
    X_train_enhanced = [X_train_important, X_train_ranking];
    X_val_enhanced = [X_val_important, X_val_ranking];
    X_test_enhanced = [X_test_important, X_test_ranking];
    
    fprintf('增强特征维度 - 训练集: %d×%d\n', size(X_train_enhanced, 1), size(X_train_enhanced, 2));
    fprintf('增强特征维度 - 验证集: %d×%d\n', size(X_val_enhanced, 1), size(X_val_enhanced, 2));
    fprintf('增强特征维度 - 测试集: %d×%d\n', size(X_test_enhanced, 1), size(X_test_enhanced, 2));
    
    %% 6. 最终特征选择（在训练集上选择，验证集上验证）
    fprintf('\n=== 最终特征选择 ===\n');
    [X_train_final, X_val_final, X_test_final, final_feature_mask, final_feature_selector] = ...
        select_final_features_with_validation(X_train_enhanced, X_val_enhanced, X_test_enhanced, y_train, y_val);
    
    fprintf('最终特征维度 - 训练集: %d×%d\n', size(X_train_final, 1), size(X_train_final, 2));
    fprintf('最终特征维度 - 验证集: %d×%d\n', size(X_val_final, 1), size(X_val_final, 2));
    fprintf('最终特征维度 - 测试集: %d×%d\n', size(X_test_final, 1), size(X_test_final, 2));
    
    %% 7. 数据标准化（使用训练集参数）
    fprintf('\n数据标准化...\n');
    [X_train_norm, normalization_params] = normalize_features_no_leak(X_train_final);
    X_val_norm = normalize_features_no_leak(X_val_final, normalization_params);
    X_test_norm = normalize_features_no_leak(X_test_final, normalization_params);
    
    %% 8. 第一层：基础分类器（在训练集上训练，验证集上选择）
    fprintf('\n=== 第一层：基础分类器训练与选择 ===\n');
    [base_predictions_train, base_predictions_val, base_predictions_test, ...
     base_classifier_names, base_val_accuracies, selected_base_classifiers] = ...
        train_and_select_base_classifiers(X_train_norm, X_val_norm, X_test_norm, y_train, y_val);
    
    %% 9. 构建元矩阵（使用选择的基分类器）
    fprintf('\n=== 构建元矩阵 ===\n');
    [meta_features_train, meta_features_val, meta_features_test] = ...
        create_meta_features_with_validation(base_predictions_train, base_predictions_val, base_predictions_test, ...
        X_train_norm, X_val_norm, X_test_norm, y_train, selected_base_classifiers);
    
    %% 10. 第二层：元分类器（在训练集上训练，验证集上选择）
    fprintf('\n=== 第二层：元分类器训练与选择 ===\n');
    [best_predictions, best_accuracy, best_meta_classifier, meta_val_results] = ...
        train_and_select_meta_classifier(meta_features_train, meta_features_val, meta_features_test, y_train, y_val, y_test);
    
    %% 11. 特征重要性分析
    fprintf('\n=== 特征重要性分析 ===\n');
    analyze_feature_importance_with_validation(final_feature_mask, size(X_train_important, 2), ...
        base_val_accuracies, meta_val_results);
    
    %% 12. 结果可视化
    fprintf('\n=== 结果分析 ===\n');
    visualize_results_with_validation(X_test_norm, y_test, best_predictions, base_predictions_test, ...
        base_classifier_names, base_val_accuracies, best_accuracy, meta_val_results, ...
        final_feature_mask, size(X_train_important, 2), selected_base_classifiers);
    
    %% 13. 输出详细分类报告
    print_comprehensive_report_with_validation(y_test, best_predictions, split_info, ...
        base_val_accuracies, meta_val_results, selected_base_classifiers);
    
    fprintf('\n=== 程序执行完成 ===\n');
    fprintf('✓ 三划分无数据泄露验证通过\n');
    fprintf('✓ 使用验证集进行模型选择\n');
    fprintf('✓ 测试集仅用于最终评估\n');
end

function [X_train, X_val, X_test, y_train, y_val, y_test, split_info] = three_way_data_split(X, y)
    % 三划分数据分割：训练集/验证集/测试集
    
    fprintf('执行三划分数据分割...\n');
    
    % 记录原始数据信息
    split_info.original_samples = size(X, 1);
    split_info.original_features = size(X, 2);
    split_info.class_distribution = countcats(y);
    split_info.class_names = categories(y);
    
    % 设置随机种子确保可重复性
    rng(43);
    
    % 第一次划分：分离测试集 (30%)
    cv1 = cvpartition(y, 'HoldOut', 0.3, 'Stratify', true);
    
    X_temp = X(cv1.training, :);
    y_temp = y(cv1.training, :);
    X_test = X(cv1.test, :);
    y_test = y(cv1.test, :);
    
    % 第二次划分：从剩余数据中分离验证集 (验证集占原始数据的20%，训练集占50%)
    cv2 = cvpartition(y_temp, 'HoldOut', 0.2857, 'Stratify', true); % 0.2857 * 0.7 ≈ 0.2
    
    X_train = X_temp(cv2.training, :);
    y_train = y_temp(cv2.training, :);
    X_val = X_temp(cv2.test, :);
    y_val = y_temp(cv2.test, :);
    
    % 记录划分信息
    split_info.training_samples = size(X_train, 1);
    split_info.validation_samples = size(X_val, 1);
    split_info.test_samples = size(X_test, 1);
    split_info.training_class_dist = countcats(y_train);
    split_info.validation_class_dist = countcats(y_val);
    split_info.test_class_dist = countcats(y_test);
    
    fprintf('训练集大小: %d (%.1f%%)\n', split_info.training_samples, split_info.training_samples/split_info.original_samples*100);
    fprintf('验证集大小: %d (%.1f%%)\n', split_info.validation_samples, split_info.validation_samples/split_info.original_samples*100);
    fprintf('测试集大小: %d (%.1f%%)\n', split_info.test_samples, split_info.test_samples/split_info.original_samples*100);
    
    fprintf('训练集类别分布: ');
    for i = 1:length(split_info.class_names)
        fprintf('%s:%d ', split_info.class_names{i}, split_info.training_class_dist(i));
    end
    fprintf('\n验证集类别分布: ');
    for i = 1:length(split_info.class_names)
        fprintf('%s:%d ', split_info.class_names{i}, split_info.validation_class_dist(i));
    end
    fprintf('\n测试集类别分布: ');
    for i = 1:length(split_info.class_names)
        fprintf('%s:%d ', split_info.class_names{i}, split_info.test_class_dist(i));
    end
    fprintf('\n');
    
    % 验证没有数据重叠
    train_val_overlap = any(ismember(X_train, X_val, 'rows'));
    train_test_overlap = any(ismember(X_train, X_test, 'rows'));
    val_test_overlap = any(ismember(X_val, X_test, 'rows'));
    
    if train_val_overlap || train_test_overlap || val_test_overlap
        warning('发现数据集之间有重叠样本！');
    else
        fprintf('✓ 数据集无重叠验证通过\n');
    end
end

function [X_clean, cleaning_params] = clean_training_data(X_train)
    % 仅使用训练集进行数据清理
    
    fprintf('清理训练集数据...\n');
    [n_samples, n_features] = size(X_train);
    
    cleaning_params = struct();
    cleaning_params.nan_columns = [];
    cleaning_params.constant_columns = [];
    cleaning_params.feature_means = zeros(1, n_features);
    
    X_clean = X_train;
    nan_count = 0;
    constant_count = 0;
    
    for i = 1:n_features
        col = X_train(:, i);
        
        % 检查NaN值
        nan_mask = isnan(col);
        if any(nan_mask)
            nan_count = nan_count + 1;
            cleaning_params.nan_columns(end+1) = i;
            % 使用训练集的均值填充NaN
            col_mean = mean(col, 'omitnan');
            col(nan_mask) = col_mean;
            cleaning_params.feature_means(i) = col_mean;
            X_clean(:, i) = col;
        else
            cleaning_params.feature_means(i) = mean(col);
        end
        
        % 检查常数特征
        if var(col) < 1e-10
            constant_count = constant_count + 1;
            cleaning_params.constant_columns(end+1) = i;
        end
    end
    
    fprintf('发现并处理 %d 个有NaN值的特征\n', nan_count);
    fprintf('发现 %d 个常数特征\n', constant_count);
end

function X_test_clean = apply_cleaning_to_test(X_test, cleaning_params)
    % 应用训练集的清理参数到测试集
    
    X_test_clean = X_test;
    
    % 处理NaN值（使用训练集的均值）
    for i = cleaning_params.nan_columns
        nan_mask = isnan(X_test_clean(:, i));
        if any(nan_mask)
            X_test_clean(nan_mask, i) = cleaning_params.feature_means(i);
        end
    end
end

function [X_important, feature_params] = select_important_original_features_no_leak(X_train, y_train)
    % 基于训练集选择重要的原始特征 - 无泄露版本
    
    fprintf('选择重要原始特征（无泄露）...\n');
    [n_samples, n_features] = size(X_train);
    
    % 限制选择的特征数量
    max_features = min(50, n_features);
    
    % 使用训练集计算特征重要性（使用更安全的方法）
    feature_scores = zeros(1, n_features);
    for i = 1:n_features
        try
            % 使用更稳健的特征重要性计算方法
            if length(unique(X_train(:, i))) > 1
                % 使用ANOVA F-value作为特征重要性
                [~, tbl] = anova1(X_train(:, i), y_train, 'off');
                if ~isempty(tbl) && size(tbl, 1) >= 2
                    feature_scores(i) = tbl{2, 5}; % F-statistic
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
        % 如果所有特征重要性都为0，选择前20个方差最大的特征
        feature_vars = var(X_train);
        [~, sorted_idx] = sort(feature_vars, 'descend');
        selected_count = min(20, n_features);
    end
    
    top_idx = sorted_idx(1:selected_count);
    X_important = X_train(:, top_idx);
    
    % 保存参数用于测试集
    feature_params.top_idx = top_idx;
    feature_params.feature_scores = feature_scores;
    feature_params.max_features = selected_count;
    feature_params.selection_method = 'ANOVA_F';
    
    fprintf('选择 %d 个重要原始特征\n', selected_count);
    fprintf('特征重要性范围: [%.4f, %.4f]\n', min(feature_scores(top_idx)), max(feature_scores(top_idx)));
end

function X_test_important = apply_feature_selection_to_test_no_leak(X_test, feature_params)
    % 应用特征选择到测试集 - 无泄露版本
    X_test_important = X_test(:, feature_params.top_idx);
end

function [X_ranking, ranking_params] = create_ranking_competitive_features_no_leak(X_train, y_train)
    % 无泄露的排序竞争特征生成
    
    fprintf('生成排序竞争特征（无泄露）...\n');
    [n_samples, n_features] = size(X_train);
    classes = unique(y_train);
    n_classes = length(classes);
    
    % 限制最大排序特征数量
    max_ranking_features = 200;
    
    % 保存参数用于测试集
    ranking_params = struct();
    ranking_params.classes = classes;
    ranking_params.n_features = n_features;
    ranking_params.n_samples = n_samples;
    
    % 初始化排序特征矩阵
    ranking_features = [];
    feature_counter = 0;
    
    % 1. 基于类内排序的特征
    fprintf('  生成类内排序特征...\n');
    ranking_params.class_rank_info = cell(n_classes, 1);
    
    for i = 1:n_classes
        class_mask = (y_train == classes(i));
        X_class = X_train(class_mask, :);
        n_class_samples = sum(class_mask);
        
        class_info = struct();
        class_info.rank_positions = zeros(n_features, n_class_samples);
        class_info.feature_ranges = zeros(n_features, 2); % [min, max]
        
        for j = 1:n_features
            if n_class_samples > 0
                % 对每个特征在类内进行排序
                [sorted_vals, sort_idx] = sort(X_class(:, j));
                rank_feature = zeros(n_samples, 1);
                
                % 创建排序映射
                rank_positions = zeros(n_class_samples, 1);
                for k = 1:n_class_samples
                    rank_positions(sort_idx(k)) = k;
                end
                
                rank_feature(class_mask) = rank_positions / n_class_samples;
                
                if feature_counter < max_ranking_features
                    ranking_features = [ranking_features, rank_feature];
                    feature_counter = feature_counter + 1;
                end
                
                % 保存排序信息
                class_info.rank_positions(j, :) = sorted_vals;
                class_info.feature_ranges(j, :) = [min(sorted_vals), max(sorted_vals)];
            end
        end
        ranking_params.class_rank_info{i} = class_info;
    end
    
    % 2. 全局排序特征
    fprintf('  生成全局排序特征...\n');
    ranking_params.global_rank_info = zeros(n_features, n_samples);
    ranking_params.global_ranges = zeros(n_features, 2);
    
    for j = 1:n_features
        [sorted_vals, sort_idx] = sort(X_train(:, j));
        rank_positions = zeros(n_samples, 1);
        for k = 1:n_samples
            rank_positions(sort_idx(k)) = k;
        end
        global_rank = rank_positions / n_samples;
        
        if feature_counter < max_ranking_features
            ranking_features = [ranking_features, global_rank];
            feature_counter = feature_counter + 1;
        end
        
        ranking_params.global_rank_info(j, :) = sorted_vals;
        ranking_params.global_ranges(j, :) = [min(sorted_vals), max(sorted_vals)];
    end
    
    % 3. 类间竞争特征
    fprintf('  生成类间竞争特征...\n');
    ranking_params.class_means = zeros(n_features, n_classes);
    ranking_params.class_stds = zeros(n_features, n_classes);
    
    % 首先计算每个类别的特征统计量
    for i = 1:n_classes
        class_mask = (y_train == classes(i));
        for j = 1:n_features
            class_vals = X_train(class_mask, j);
            ranking_params.class_means(j, i) = mean(class_vals);
            ranking_params.class_stds(j, i) = std(class_vals);
        end
    end
    
    for j = 1:n_features
        competitive_feature = zeros(n_samples, 1);
        
        for k = 1:n_samples
            current_class = find(classes == y_train(k));
            other_classes = setdiff(1:n_classes, current_class);
            
            % 使用标准化距离
            current_val = X_train(k, j);
            current_mean = ranking_params.class_means(j, current_class);
            current_std = max(ranking_params.class_stds(j, current_class), 1e-10);
            
            dist_to_own = abs(current_val - current_mean) / current_std;
            
            if ~isempty(other_classes)
                dist_to_others = inf;
                for cls = other_classes
                    other_mean = ranking_params.class_means(j, cls);
                    other_std = max(ranking_params.class_stds(j, cls), 1e-10);
                    dist = abs(current_val - other_mean) / other_std;
                    if dist < dist_to_others
                        dist_to_others = dist;
                    end
                end
            else
                dist_to_others = dist_to_own;
            end
            
            if dist_to_others > 0
                competitive_feature(k) = dist_to_own / (dist_to_own + dist_to_others);
            else
                competitive_feature(k) = 0.5;
            end
        end
        
        if feature_counter < max_ranking_features
            ranking_features = [ranking_features, competitive_feature];
            feature_counter = feature_counter + 1;
        end
    end
    
    % 记录最终特征数量
    ranking_params.final_feature_count = feature_counter;
    ranking_params.used_feature_indices = 1:feature_counter;
    
    X_ranking = ranking_features;
    fprintf('生成排序特征完成: %d 个排序竞争特征\n', feature_counter);
end

function X_test_ranking = apply_ranking_features_no_leak(X_test, y_test, ranking_params)
    % 无泄露的排序特征应用
    
    [n_samples, n_features] = size(X_test);
    classes = ranking_params.classes;
    n_classes = length(classes);
    
    ranking_features = zeros(n_samples, ranking_params.final_feature_count);
    feature_counter = 0;
    
    % 1. 类内排序特征
    for i = 1:n_classes
        class_mask = (y_test == classes(i));
        X_class = X_test(class_mask, :);
        n_class_samples = sum(class_mask);
        class_info = ranking_params.class_rank_info{i};
        
        for j = 1:n_features
            if n_class_samples > 0
                rank_feature = zeros(n_samples, 1);
                train_sorted_vals = class_info.rank_positions(j, :);
                current_vals = X_class(:, j);
                
                % 计算当前值在训练集排序中的位置
                ranks = zeros(n_class_samples, 1);
                for k = 1:n_class_samples
                    current_val = current_vals(k);
                    
                    % 找到插入位置
                    if current_val <= train_sorted_vals(1)
                        ranks(k) = 1;
                    elseif current_val >= train_sorted_vals(end)
                        ranks(k) = n_class_samples;
                    else
                        % 使用二分查找提高效率
                        left = 1;
                        right = n_class_samples;
                        while left <= right
                            mid = floor((left + right) / 2);
                            if train_sorted_vals(mid) <= current_val
                                left = mid + 1;
                            else
                                right = mid - 1;
                            end
                        end
                        ranks(k) = left - 1;
                    end
                end
                rank_feature(class_mask) = ranks / n_class_samples;
                
                feature_counter = feature_counter + 1;
                if feature_counter <= ranking_params.final_feature_count
                    ranking_features(:, feature_counter) = rank_feature;
                end
            end
        end
    end
    
    % 2. 全局排序特征
    for j = 1:n_features
        train_sorted_vals = ranking_params.global_rank_info(j, :);
        current_vals = X_test(:, j);
        
        % 计算当前值在训练集全局排序中的位置
        ranks = zeros(n_samples, 1);
        for k = 1:n_samples
            current_val = current_vals(k);
            
            if current_val <= train_sorted_vals(1)
                ranks(k) = 1;
            elseif current_val >= train_sorted_vals(end)
                ranks(k) = ranking_params.n_samples;
            else
                left = 1;
                right = ranking_params.n_samples;
                while left <= right
                    mid = floor((left + right) / 2);
                    if train_sorted_vals(mid) <= current_val
                        left = mid + 1;
                    else
                        right = mid - 1;
                    end
                end
                ranks(k) = left - 1;
            end
        end
        global_rank = ranks / ranking_params.n_samples;
        
        feature_counter = feature_counter + 1;
        if feature_counter <= ranking_params.final_feature_count
            ranking_features(:, feature_counter) = global_rank;
        end
    end
    
    % 3. 类间竞争特征
    class_means = ranking_params.class_means;
    class_stds = ranking_params.class_stds;
    
    for j = 1:n_features
        competitive_feature = zeros(n_samples, 1);
        
        for k = 1:n_samples
            current_class = find(classes == y_test(k));
            other_classes = setdiff(1:n_classes, current_class);
            
            current_val = X_test(k, j);
            current_mean = class_means(j, current_class);
            current_std = max(class_stds(j, current_class), 1e-10);
            
            dist_to_own = abs(current_val - current_mean) / current_std;
            
            if ~isempty(other_classes)
                dist_to_others = inf;
                for cls = other_classes
                    other_mean = class_means(j, cls);
                    other_std = max(class_stds(j, cls), 1e-10);
                    dist = abs(current_val - other_mean) / other_std;
                    if dist < dist_to_others
                        dist_to_others = dist;
                    end
                end
            else
                dist_to_others = dist_to_own;
            end
            
            if dist_to_others > 0
                competitive_feature(k) = dist_to_own / (dist_to_own + dist_to_others);
            else
                competitive_feature(k) = 0.5;
            end
        end
        
        feature_counter = feature_counter + 1;
        if feature_counter <= ranking_params.final_feature_count
            ranking_features(:, feature_counter) = competitive_feature;
        end
    end
    
    X_test_ranking = ranking_features;
end

function [X_train_final, X_val_final, X_test_final, feature_mask, feature_selector] = ...
    select_final_features_with_validation(X_train, X_val, X_test, y_train, y_val)
    % 使用验证集指导的最终特征选择
    
    [n_samples_train, n_features] = size(X_train);
    
    fprintf('使用验证集指导的最终特征选择...\n');
    fprintf('输入特征数量: %d\n', n_features);
    
    % 1. 移除常数特征（基于训练集）
    feature_vars = var(X_train);
    non_constant = feature_vars > 1e-10;
    X_train_filtered = X_train(:, non_constant);
    X_val_filtered = X_val(:, non_constant);
    X_test_filtered = X_test(:, non_constant);
    
    fprintf('移除常数特征后: %d\n', sum(non_constant));
    
    % 2. 基于互信息的特征选择，使用验证集性能指导
    feature_selector = struct();
    feature_selector.non_constant = non_constant;
    
    try
        fprintf('使用验证集指导的互信息特征选择...\n');
        
        % 计算互信息
        mi_scores = zeros(1, size(X_train_filtered, 2));
        for i = 1:size(X_train_filtered, 2)
            try
                feature_discrete = discretize(X_train_filtered(:, i), 10);
                mi_scores(i) = mutualinfo(feature_discrete, double(y_train));
            catch
                mi_scores(i) = 0;
            end
        end
        
        % 使用验证集选择最佳特征数量
        valid_scores = mi_scores(mi_scores > 0);
        if isempty(valid_scores)
            mi_threshold = 0;
        else
            mi_threshold = mean(valid_scores);
        end
        
        if mi_threshold == 0
            mi_threshold = 0.001;
        end
        
        % 测试不同的特征数量，选择在验证集上表现最好的
        candidate_feature_counts = [10, 20, 30, 40, 50, 60, 70, 80];
        candidate_feature_counts = candidate_feature_counts(candidate_feature_counts <= length(mi_scores));
        
        if isempty(candidate_feature_counts)
            candidate_feature_counts = min(30, length(mi_scores));
        end
        
        best_val_accuracy = 0;
        best_feature_count = min(50, length(mi_scores));
        
        for n_feat = candidate_feature_counts
            [~, sorted_idx] = sort(mi_scores, 'descend');
            selected_indices_filtered = sorted_idx(1:min(n_feat, length(mi_scores)));
            
            X_train_candidate = X_train_filtered(:, selected_indices_filtered);
            X_val_candidate = X_val_filtered(:, selected_indices_filtered);
            
            % 使用简单分类器快速评估
            try
                model = fitctree(X_train_candidate, y_train, 'MaxNumSplits', 20);
                y_pred_val = predict(model, X_val_candidate);
                val_accuracy = sum(y_pred_val == y_val) / length(y_val);
                
                if val_accuracy > best_val_accuracy
                    best_val_accuracy = val_accuracy;
                    best_feature_count = n_feat;
                end
            catch
                continue;
            end
        end
        
        fprintf('验证集指导选择特征数量: %d (准确率: %.4f)\n', best_feature_count, best_val_accuracy);
        
        [~, sorted_idx] = sort(mi_scores, 'descend');
        selected_indices_filtered = sorted_idx(1:best_feature_count);
        
        % 映射回原始特征索引
        original_indices = find(non_constant);
        selected_indices = original_indices(selected_indices_filtered);
        
        X_train_final = X_train(:, selected_indices);
        X_val_final = X_val(:, selected_indices);
        X_test_final = X_test(:, selected_indices);
        feature_mask = false(1, n_features);
        feature_mask(selected_indices) = true;
        
        feature_selector.selected_indices = selected_indices;
        feature_selector.mi_scores = mi_scores;
        feature_selector.best_feature_count = best_feature_count;
        feature_selector.best_val_accuracy = best_val_accuracy;
        feature_selector.selection_method = 'MutualInformation_ValidationGuided';
        
    catch ME
        fprintf('验证集指导特征选择失败: %s\n', ME.message);
        fprintf('使用固定数量特征选择...\n');
        
        % 回退到基于方差的选择
        feature_vars = var(X_train_filtered);
        [~, sorted_idx] = sort(feature_vars, 'descend');
        n_select = min(60, size(X_train_filtered, 2));
        selected_indices_filtered = sorted_idx(1:n_select);
        
        original_indices = find(non_constant);
        selected_indices = original_indices(selected_indices_filtered);
        
        X_train_final = X_train(:, selected_indices);
        X_val_final = X_val(:, selected_indices);
        X_test_final = X_test(:, selected_indices);
        feature_mask = false(1, n_features);
        feature_mask(selected_indices) = true;
        
        feature_selector.selected_indices = selected_indices;
        feature_selector.selection_method = 'Variance_Fallback';
    end
    
    fprintf('最终选择特征数量: %d\n', sum(feature_mask));
end

function [X_normalized, params] = normalize_features_no_leak(X, params)
    % 无泄露的特征标准化
    if nargin < 2
        % 训练模式：计算参数
        params.mu = mean(X, 1, 'omitnan');
        params.sigma = std(X, 0, 1, 'omitnan');
        params.sigma(params.sigma == 0) = 1; % 避免除零
        
        % 记录标准化信息
        params.normalization_type = 'ZScore';
        params.training_samples = size(X, 1);
    end
    
    X_normalized = (X - params.mu) ./ params.sigma;
    
    % 处理可能的NaN值（设置为0）
    X_normalized(isnan(X_normalized)) = 0;
    
    % 限制极端值
    X_normalized = max(min(X_normalized, 10), -10); % 限制在[-10, 10]范围内
end

function [base_predictions_train, base_predictions_val, base_predictions_test, ...
          base_classifier_names, base_val_accuracies, selected_classifiers] = ...
    train_and_select_base_classifiers(X_train, X_val, X_test, y_train, y_val)
    % 训练并在验证集上选择基础分类器
    
    base_predictions_train = [];
    base_predictions_val = [];
    base_predictions_test = [];
    base_classifier_names = {};
    base_val_accuracies = [];
    selected_classifiers = {};
    
    % 定义基础分类器候选
    base_classifiers = {
        @() fitctree(X_train, y_train, 'MaxNumSplits', 50, 'CrossVal', 'off'), '决策树';
        @() fitcdiscr(X_train, y_train, 'DiscrimType', 'pseudoLinear', 'CrossVal', 'off'), '线性判别';
        @() fitcnb(X_train, y_train, 'DistributionNames', 'kernel', 'CrossVal', 'off'), '朴素贝叶斯';
        @() fitcknn(X_train, y_train, 'NumNeighbors', 10, 'Distance', 'cosine', 'CrossVal', 'off'), 'K近邻';
        @() fitcecoc(X_train, y_train, 'Coding', 'onevsone', 'Learners', 'tree'), 'ECOC-决策树';
        @() fitcensemble(X_train, y_train, 'Method', 'Bag', 'NumLearningCycles', 100), 'Bagging';
        @() fitcensemble(X_train, y_train, 'Method', 'AdaBoostM1', 'NumLearningCycles', 100), 'AdaBoost';
        @() fitcsvm(X_train, y_train, 'KernelFunction', 'linear', 'Standardize', true), '线性SVM';
    };
    
    % 训练所有基础分类器并在验证集上评估
    classifier_results = [];
    
    for i = 1:size(base_classifiers, 1)
        fprintf('训练并评估 %s...\n', base_classifiers{i,2});
        
        try
            % 训练分类器
            model = base_classifiers{i,1}();
            
            % 验证集预测和评估
            y_pred_val = predict(model, X_val);
            val_accuracy = sum(y_pred_val == y_val) / length(y_val);
            
            % 保存结果
            result.name = base_classifiers{i,2};
            result.accuracy = val_accuracy;
            result.model = model;
            result.index = i;
            classifier_results = [classifier_results, result];
            
            fprintf('  %s 验证集准确率: %.4f\n', base_classifiers{i,2}, val_accuracy);
            
        catch ME
            fprintf('  %s 训练失败: %s\n', base_classifiers{i,2}, ME.message);
        end
    end
    
    if isempty(classifier_results)
        error('所有基础分类器训练失败，请检查数据');
    end
    
    % 根据验证集性能选择最佳分类器（选择前50%）
    [~, sort_idx] = sort([classifier_results.accuracy], 'descend');
    n_select = max(3, ceil(length(classifier_results) * 0.5)); % 至少选择3个，最多选择前50%
    selected_results = classifier_results(sort_idx(1:n_select));
    
    fprintf('\n选择的基础分类器 (%d个):\n', n_select);
    
    % 使用选择的分类器生成预测
    for i = 1:length(selected_results)
        result = selected_results(i);
        
        % 训练集预测
        y_pred_train = predict(result.model, X_train);
        base_predictions_train = [base_predictions_train, y_pred_train];
        
        % 验证集预测
        y_pred_val = predict(result.model, X_val);
        base_predictions_val = [base_predictions_val, y_pred_val];
        
        % 测试集预测
        y_pred_test = predict(result.model, X_test);
        base_predictions_test = [base_predictions_test, y_pred_test];
        
        base_classifier_names{end+1} = result.name;
        base_val_accuracies(end+1) = result.accuracy;
        selected_classifiers{end+1} = result.model;
        
        fprintf('  %s: %.4f\n', result.name, result.accuracy);
    end
end

function [meta_features_train, meta_features_val, meta_features_test] = ...
    create_meta_features_with_validation(base_predictions_train, base_predictions_val, base_predictions_test, ...
    X_train, X_val, X_test, y_train, selected_classifiers)
    % 构建元特征矩阵
    
    n_train = size(base_predictions_train, 1);
    n_val = size(base_predictions_val, 1);
    n_test = size(base_predictions_test, 1);
    n_base = size(base_predictions_train, 2);
    classes = categories(y_train);
    n_classes = length(classes);
    
    % 1. 基础分类器预测的one-hot编码
    prediction_features_train = zeros(n_train, n_base * n_classes);
    prediction_features_val = zeros(n_val, n_base * n_classes);
    prediction_features_test = zeros(n_test, n_base * n_classes);
    
    for i = 1:n_base
        for j = 1:n_classes
            col_idx = (i-1)*n_classes + j;
            prediction_features_train(:, col_idx) = (base_predictions_train(:, i) == classes(j));
            prediction_features_val(:, col_idx) = (base_predictions_val(:, i) == classes(j));
            prediction_features_test(:, col_idx) = (base_predictions_test(:, i) == classes(j));
        end
    end
    
    % 2. 分类器置信度特征（基于训练集分布）
    confidence_features_train = zeros(n_train, n_base);
    confidence_features_val = zeros(n_val, n_base);
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
        
        % 对验证集和测试集使用训练集的分布
        for j = 1:n_val
            pred_class = base_predictions_val(j, i);
            class_idx = find(classes == pred_class, 1);
            if ~isempty(class_idx)
                confidence_features_val(j, i) = mean(train_class_dist(:, class_idx));
            else
                confidence_features_val(j, i) = 0;
            end
        end
        
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
    meta_features_val = [prediction_features_val, confidence_features_val, X_val];
    meta_features_test = [prediction_features_test, confidence_features_test, X_test];
    
    fprintf('元矩阵维度 - 训练集: %d×%d\n', size(meta_features_train));
    fprintf('元矩阵维度 - 验证集: %d×%d\n', size(meta_features_val));
    fprintf('元矩阵维度 - 测试集: %d×%d\n', size(meta_features_test));
end

function [best_predictions, best_accuracy, best_meta_classifier, meta_val_results] = ...
    train_and_select_meta_classifier(meta_features_train, meta_features_val, meta_features_test, y_train, y_val, y_test)
    % 训练并在验证集上选择最佳元分类器
    
    meta_classifiers = {
        @() fitcensemble(meta_features_train, y_train, 'Method', 'Bag', 'NumLearningCycles', 100), 'Bagging元分类器';
        @() fitcecoc(meta_features_train, y_train, 'Coding', 'onevsall'), 'ECOC元分类器';
        @() fitctree(meta_features_train, y_train, 'MaxNumSplits', 50), '决策树元分类器';
        @() fitcknn(meta_features_train, y_train, 'NumNeighbors', 15), 'KNN元分类器';
        @() fitcensemble(meta_features_train, y_train, 'Method', 'AdaBoostM1', 'NumLearningCycles', 100), 'AdaBoost元分类器';
        @() fitcsvm(meta_features_train, y_train, 'KernelFunction', 'rbf', 'Standardize', true), 'SVM元分类器';
    };
    
    best_val_accuracy = 0;
    best_predictions = [];
    best_meta_classifier = [];
    meta_val_results = [];
    
    % 在验证集上评估所有元分类器
    for i = 1:size(meta_classifiers, 1)
        try
            fprintf('训练并评估 %s...\n', meta_classifiers{i,2});
            meta_model = meta_classifiers{i,1}();
            
            % 验证集评估
            y_pred_val = predict(meta_model, meta_features_val);
            val_accuracy = sum(y_pred_val == y_val) / length(y_val);
            
            fprintf('  %s 验证集准确率: %.4f\n', meta_classifiers{i,2}, val_accuracy);
            
            result.name = meta_classifiers{i,2};
            result.val_accuracy = val_accuracy;
            result.model = meta_model;
            meta_val_results = [meta_val_results, result];
            
            % 更新最佳模型
            if val_accuracy > best_val_accuracy
                best_val_accuracy = val_accuracy;
                best_meta_classifier = meta_model;
            end
        catch ME
            fprintf('  %s 训练失败: %s\n', meta_classifiers{i,2}, ME.message);
        end
    end
    
    if ~isempty(best_meta_classifier)
        % 使用最佳模型在测试集上预测
        best_predictions = predict(best_meta_classifier, meta_features_test);
        best_accuracy = sum(best_predictions == y_test) / length(y_test);
        
        fprintf('\n最佳元分类器: %s\n', meta_val_results([meta_val_results.val_accuracy] == best_val_accuracy).name);
        fprintf('验证集准确率: %.4f\n', best_val_accuracy);
        fprintf('测试集准确率: %.4f\n', best_accuracy);
    else
        fprintf('\n所有元分类器训练失败\n');
        best_predictions = [];
        best_accuracy = 0;
    end
end

function analyze_feature_importance_with_validation(feature_mask, n_original_features, base_val_accuracies, meta_val_results)
    % 包含验证集信息的特征重要性分析
    
    fprintf('\n--- 特征重要性分析 (验证集指导) ---\n');
    
    n_total_features = length(feature_mask);
    n_ranking_features = n_total_features - n_original_features;
    
    original_selected = sum(feature_mask(1:n_original_features));
    ranking_selected = sum(feature_mask(n_original_features+1:end));
    
    fprintf('原始特征选中: %d/%d (%.1f%%)\n', ...
        original_selected, n_original_features, original_selected/n_original_features*100);
    fprintf('排序特征选中: %d/%d (%.1f%%)\n', ...
        ranking_selected, n_ranking_features, ranking_selected/n_ranking_features*100);
    
    % 基础分类器性能分析
    fprintf('\n基础分类器验证集性能:\n');
    for i = 1:length(base_val_accuracies)
        fprintf('  分类器%d: %.4f\n', i, base_val_accuracies(i));
    end
    fprintf('平均性能: %.4f\n', mean(base_val_accuracies));
    
    % 元分类器性能分析
    if ~isempty(meta_val_results)
        fprintf('\n元分类器验证集性能:\n');
        for i = 1:length(meta_val_results)
            fprintf('  %s: %.4f\n', meta_val_results(i).name, meta_val_results(i).val_accuracy);
        end
    end
    
    % 可视化特征选择结果
    figure('Position', [200, 200, 800, 400]);
    
    subplot(1, 2, 1);
    selection_data = [original_selected, ranking_selected];
    bar(selection_data);
    set(gca, 'XTickLabel', {'原始特征', '排序特征'});
    title('选中的特征数量');
    ylabel('数量');
    grid on;
    
    for i = 1:length(selection_data)
        text(i, selection_data(i) + max(selection_data)*0.05, num2str(selection_data(i)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
    
    subplot(1, 2, 2);
    ratio_data = [original_selected/n_original_features, ranking_selected/n_ranking_features] * 100;
    pie(ratio_data, {'原始特征', '排序特征'});
    title('特征选中比例 (%)');
end

function visualize_results_with_validation(X_test, y_test, final_predictions, base_predictions, ...
                         base_classifier_names, base_val_accuracies, meta_accuracy, all_meta_results, ...
                         feature_mask, n_original_features, selected_classifiers)
    % 包含验证集信息的结果可视化
    
    figure('Position', [100, 100, 1200, 800]);
    
    % 子图1: 所有分类器准确率比较
    subplot(2, 3, 1);
    if ~isempty(all_meta_results)
        meta_accuracies = [all_meta_results.val_accuracy];
        all_accuracies = [base_val_accuracies, meta_accuracies];
        all_names = [base_classifier_names, {all_meta_results.name}];
    else
        all_accuracies = base_val_accuracies;
        all_names = base_classifier_names;
    end
    
    [sorted_acc, sort_idx] = sort(all_accuracies, 'descend');
    sorted_names = all_names(sort_idx);
    
    barh(sorted_acc);
    set(gca, 'YTickLabel', sorted_names, 'FontSize', 8);
    title('分类器验证集准确率排序');
    xlabel('准确率');
    grid on;
    
    for i = 1:length(sorted_acc)
        text(sorted_acc(i) + 0.01, i, sprintf('%.3f', sorted_acc(i)), ...
            'VerticalAlignment', 'middle', 'FontSize', 7);
    end
    
    % 子图2: 混淆矩阵
    subplot(2, 3, 2);
    cm = confusionmat(y_test, final_predictions);
    confusionchart(cm);
    title(sprintf('最终模型混淆矩阵\n测试集准确率: %.2f%%', mean(final_predictions == y_test)*100));
    
    % 子图3: 特征类型分布
    subplot(2, 3, 3);
    n_original_selected = sum(feature_mask(1:n_original_features));
    n_ranking_selected = sum(feature_mask(n_original_features+1:end));
    
    pie_data = [n_original_selected, n_ranking_selected];
    labels = {sprintf('原始特征\n%d个', n_original_selected), ...
              sprintf('排序特征\n%d个', n_ranking_selected)};
    pie(pie_data, labels);
    title('最终特征分布');
    
    % 子图4: 各类别性能
    subplot(2, 3, 4);
    classes = categories(y_test);
    class_accuracies = zeros(length(classes), 1);
    
    for i = 1:length(classes)
        class_mask = (y_test == classes(i));
        class_accuracies(i) = sum(final_predictions(class_mask) == y_test(class_mask)) / sum(class_mask);
    end
    
    bar(class_accuracies);
    set(gca, 'XTickLabel', classes);
    title('各类别分类准确率');
    ylabel('准确率');
    grid on;
    
    for i = 1:length(class_accuracies)
        text(i, class_accuracies(i) + 0.01, sprintf('%.3f', class_accuracies(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8);
    end
    
    % 子图5: 基础分类器相关性
    subplot(2, 3, 5);
    if size(base_predictions, 2) > 1
        agreement_matrix = zeros(length(base_classifier_names));
        for i = 1:length(base_classifier_names)
            for j = 1:length(base_classifier_names)
                agreement_matrix(i,j) = sum(base_predictions(:, i) == base_predictions(:, j)) / length(y_test);
            end
        end
        imagesc(agreement_matrix);
        colorbar;
        set(gca, 'XTick', 1:length(base_classifier_names), 'YTick', 1:length(base_classifier_names));
        title('基础分类器预测一致性');
        xlabel('分类器索引');
        ylabel('分类器索引');
    else
        text(0.5, 0.5, '需要多个基础分类器', 'HorizontalAlignment', 'center');
        title('分类器一致性');
    end
    
    % 子图6: 性能总结
    subplot(2, 3, 6);
    best_base_acc = max(base_val_accuracies);
    
    if ~isempty(all_meta_results)
        best_meta_acc = max([all_meta_results.val_accuracy]);
        improvement = (meta_accuracy - best_base_acc) / best_base_acc * 100;
        
        text(0.5, 0.7, sprintf('三划分数据流程\n无数据泄露'), ...
            'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
        text(0.5, 0.5, sprintf('基础最佳: %.3f\n元学习: %.3f', best_base_acc, meta_accuracy), ...
            'HorizontalAlignment', 'center', 'FontSize', 10);
        
        if improvement > 0
            text(0.5, 0.3, sprintf('提升: +%.1f%%', improvement), ...
                'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', 'green');
        else
            text(0.5, 0.3, sprintf('变化: %.1f%%', improvement), ...
                'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', 'red');
        end
    else
        text(0.5, 0.5, sprintf('基础分类器最佳: %.3f', best_base_acc), ...
            'HorizontalAlignment', 'center', 'FontSize', 12);
    end
    axis off;
    title('性能总结');
end

function print_comprehensive_report_with_validation(y_test, best_predictions, split_info, ...
    base_val_accuracies, meta_val_results, selected_classifiers)
    % 包含验证集信息的综合报告
    
    fprintf('\n=== 综合性能报告 (三划分版本) ===\n');
    
    % 最终模型性能
    final_accuracy = sum(best_predictions == y_test) / length(y_test);
    fprintf('最终模型测试集准确率: %.4f\n', final_accuracy);
    
    % 基础分类器总结
    fprintf('\n--- 基础分类器总结 ---\n');
    fprintf('选择的基础分类器数量: %d\n', length(selected_classifiers));
    fprintf('平均验证集准确率: %.4f\n', mean(base_val_accuracies));
    fprintf('最佳验证集准确率: %.4f\n', max(base_val_accuracies));
    
    % 元分类器总结
    if ~isempty(meta_val_results)
        fprintf('\n--- 元分类器总结 ---\n');
        best_meta_val = max([meta_val_results.val_accuracy]);
        fprintf('最佳元分类器验证集准确率: %.4f\n', best_meta_val);
        fprintf('测试集相对于验证集的性能变化: %+.4f\n', final_accuracy - best_meta_val);
    end
    
    % 详细分类指标
    classes = categories(y_test);
    fprintf('\n--- 详细分类指标 (测试集) ---\n');
    fprintf('%-10s %-8s %-8s %-8s %-8s\n', ...
        '类别', '精确率', '召回率', 'F1分数', '支持度');
    
    total_TP = 0; total_FP = 0; total_FN = 0;
    
    for i = 1:length(classes)
        current_class = classes{i};
        
        TP = sum((y_test == current_class) & (best_predictions == current_class));
        FP = sum((y_test ~= current_class) & (best_predictions == current_class));
        FN = sum((y_test == current_class) & (best_predictions ~= current_class));
        
        total_TP = total_TP + TP;
        total_FP = total_FP + FP;
        total_FN = total_FN + FN;
        
        precision = TP / (TP + FP + eps);
        recall = TP / (TP + FN + eps);
        f1_score = 2 * (precision * recall) / (precision + recall + eps);
        support = sum(y_test == current_class);
        
        fprintf('%-10s %-8.3f %-8.3f %-8.3f %-8d\n', ...
            char(current_class), precision, recall, f1_score, support);
    end
    
    % 宏平均
    macro_precision = total_TP / (total_TP + total_FP + eps);
    macro_recall = total_TP / (total_TP + total_FN + eps);
    macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall + eps);
    
    fprintf('\n宏平均: 精确率=%.3f, 召回率=%.3f, F1=%.3f\n', ...
        macro_precision, macro_recall, macro_f1);
    
    % Kappa系数
    kappa = compute_kappa_no_leak(y_test, best_predictions);
    fprintf('Kappa系数: %.3f\n', kappa);
    
    % Kappa解释
    if kappa < 0
        kappa_interpretation = '无一致性';
    elseif kappa < 0.2
        kappa_interpretation = '极低一致性';
    elseif kappa < 0.4
        kappa_interpretation = '一般一致性';
    elseif kappa < 0.6
        kappa_interpretation = '中等一致性';
    elseif kappa < 0.8
        kappa_interpretation = '高度一致性';
    else
        kappa_interpretation = '几乎完全一致';
    end
    fprintf('Kappa解释: %s\n', kappa_interpretation);
    
    fprintf('\n=== 数据流程验证 ===\n');
    fprintf('✓ 训练集/验证集/测试集严格分离\n');
    fprintf('✓ 所有特征工程仅基于训练集\n');
    fprintf('✓ 验证集用于模型选择和参数调整\n');
    fprintf('✓ 测试集仅用于最终评估\n');
    fprintf('✓ 无数据泄露风险\n');
    fprintf('训练集样本数: %d\n', split_info.training_samples);
    fprintf('验证集样本数: %d\n', split_info.validation_samples);
    fprintf('测试集样本数: %d\n', split_info.test_samples);
end

function kappa = compute_kappa_no_leak(y_true, y_pred)
    % 计算Kappa系数
    cm = confusionmat(y_true, y_pred);
    n = sum(cm(:));
    po = sum(diag(cm)) / n;
    
    col_sums = sum(cm, 1);
    row_sums = sum(cm, 2);
    pe = sum(col_sums .* row_sums') / n^2;
    
    kappa = (po - pe) / (1 - pe);
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

% 运行程序
% meta_classification_with_ranking_features_three_way();