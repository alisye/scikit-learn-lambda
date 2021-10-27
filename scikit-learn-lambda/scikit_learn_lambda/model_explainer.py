class LinearExplainer(object):
    def __init__(self, model):
        self.model = self._create_model_object(model)

    def explain(self, features):
        model = self.model
        f = {n: f for n, f in enumerate(features)}
        feature_weights = [0.0] * len(features)

        for index, feature in f.items():
            c = model['coefficients'][index]
            feature_weights[index] = c * feature

        return feature_weights

    @staticmethod
    def _create_model_object(lr_model):
        model_dump = {}

        if lr_model.classes_.shape[0] == 2:
            coefs = lr_model.coef_[0].tolist()
            coefs_nonzero = [(n, coef) for n, coef in enumerate(coefs) if coef != 0]

            model_dump['coefficients'] = {
                n: item[1] for n, item in enumerate(coefs_nonzero)
            }
        else:
            model_dump['coefficents'] = lr_model.coef_.tolist()

        return model_dump
