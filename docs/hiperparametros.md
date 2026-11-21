# Espaços de busca de hiperparâmetros

Busca aleatória (Bergstra; Bengio, 2012) com 40 combinações por modelo, CV 5-dobras **só no treino**, critério MAE.
Os limites seguem as faixas recomendadas por estudos de "tunabilidade", que estimam, sobre muitos conjuntos de dados,
quais hiperparâmetros importam e em que faixas o ótimo costuma cair (Probst; Boulesteix; Bischl, 2019), e a revisão
específica de Random Forest (Probst; Wright; Boulesteix, 2019). Com ~480 escolas de treino e ~360 preditores, os
limites privilegiam regularização forte (árvores rasas, folhas mínimas maiores, subamostragem de colunas).

| Modelo | Hiperparâmetro | Faixa | Justificativa |
|---|---|---|---|
| Elastic Net (proficiência) | `alpha` | log-uniforme [1e-3, 10] | cobre de quase-OLS a penalização forte |
| | `l1_ratio` | [0,05; 0,95] | do quase-Ridge ao quase-Lasso |
| Fractional logit (proporção) | `C` | log-uniforme [1e-3, 10] | inverso da penalização |
| | `l1_ratio` | [0,05; 0,95] | idem |
| Random Forest | `max_features` | {0,05; 0,1; 0,2; 0,33; 0,5; √p} | o hiperparâmetro mais "tunável" da RF (Probst et al., 2019) |
| | `min_samples_leaf` | 1–20 | controla a variância com n pequeno |
| | `max_samples` | [0,5; 1,0] | tamanho da amostra bootstrap |
| | `max_depth` | {None, 4, 8, 16} | |
| | `n_estimators` | 500 (fixo) | não é tunável: mais árvores não pioram (Probst; Boulesteix, 2018) |
| XGBoost | `learning_rate` | log-uniforme [0,005; 0,2] | tratado em conjunto com `n_estimators` |
| | `n_estimators` | 100–1000 | |
| | `max_depth` | 2–6 | árvores rasas para n pequeno |
| | `subsample` / `colsample_bytree` | [0,5; 1] / [0,2; 1] | |
| | `min_child_weight` | log-uniforme [1, 50] | |
| | `reg_lambda` / `reg_alpha` | log-uniforme [0,1; 50] / [1e-3; 10] | |
| LightGBM | `num_leaves` | 4–31 | equivalente a profundidade ≤ 5 |
| | `min_child_samples` | 5–40 | |
| | demais | como no XGBoost | |

**Referências** (conferir na fonte antes de citar):
- BERGSTRA, J.; BENGIO, Y. Random search for hyper-parameter optimization. *JMLR*, v. 13, p. 281–305, 2012.
- PROBST, P.; BOULESTEIX, A.-L.; BISCHL, B. Tunability: importance of hyperparameters of machine learning algorithms. *JMLR*, v. 20, n. 53, p. 1–32, 2019.
- PROBST, P.; WRIGHT, M. N.; BOULESTEIX, A.-L. Hyperparameters and tuning strategies for random forest. *WIREs Data Mining and Knowledge Discovery*, v. 9, n. 3, e1301, 2019.
- PROBST, P.; BOULESTEIX, A.-L. To tune or not to tune the number of trees in random forest. *JMLR*, v. 18, n. 181, p. 1–18, 2018.
- NADEAU, C.; BENGIO, Y. Inference for the generalization error. *Machine Learning*, v. 52, p. 239–281, 2003 (t corrigido usado na regra de seleção).
