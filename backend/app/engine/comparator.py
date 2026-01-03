class ComparatorEngine:
    def __init__(self, models: dict):
        self.models = models

    def run_all(self, image):
        outputs = {}
        for name, model in self.models.items():
            outputs[name] = model.run(image)
        return outputs

 