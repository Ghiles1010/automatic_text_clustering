from sklearn.decomposition import TruncatedSVD


def lsa(vectors):

    tsvd = TruncatedSVD(n_components=vectors.shape[1]-1)
    tsvd.fit(vectors)
    tsvd_var_ratios = tsvd.explained_variance_ratio_

    def select_n_components(var_ratio, goal_var: float) -> int:
        total_variance = 0.0
        n_components = 0
        for explained_variance in var_ratio:
            total_variance += explained_variance
            n_components += 1
            if total_variance >= goal_var : break
        return n_components

    best_n_components = select_n_components(tsvd_var_ratios, 0.95)
    svd_model = TruncatedSVD(n_components=best_n_components, algorithm='randomized', n_iter=100, random_state=122)
    vectors = svd_model.fit_transform(vectors)
    return vectors

