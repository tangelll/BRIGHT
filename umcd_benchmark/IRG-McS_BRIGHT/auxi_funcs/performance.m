function [tp,fp,tn,fn,fplv,fnlv,abfplv,abfnlv,pcc,kappa,imw]=performance(cm, gt, ignore_value)
gt = double(gt(:,:,1));  % 黄河一号、黄河二号数据集
cm = double(cm);

% 排除ignore_value的像素
valid_mask = gt ~= ignore_value;

% 计算有效像素的总数
[A, B] = size(gt); 
N = sum(valid_mask(:));  % 仅计算有效像素

Nu = sum(gt(:) == 0 & valid_mask(:));  % 无变化的有效像素
Nc = sum(gt(:) ~= 0 & valid_mask(:));  % 变化的有效像素

% 初始化结果变量
imw = zeros(A, B);  % 错误观察图
im = cm - gt;

% 计算各种指标：注意我们仅对有效像素进行评估
fp = sum((im(:) > 0) & valid_mask(:));
fn = sum((im(:) < 0) & valid_mask(:));
tp = Nc - fn;
tn = Nu - fp;

fplv = fp / N;  % 错误检出率
fnlv = fn / N;  % 漏检率
abfplv = fp / Nu;  % 错检的相对比例
abfnlv = fn / Nc;  % 漏检的相对比例
pcc = 1 - fplv - fnlv;  % 正确率

% 计算Kappa系数
pra = (tp + tn) / N;  % 实际一致率
pre = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (N^2);  % 期望一致率
kappa = (pra - pre) / (1 - pre);  % Kappa系数

end