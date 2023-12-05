class Review:
    SLICE_MARGIN = 8
    text = ""
    label = ""

    def __init__(self, data):
        self.text = self.get_text_from_data(data)
        self.label = self.get_label_from_data(data)

    def get_text_from_data(self, data):
        return data[:self.SLICE_MARGIN]

    def get_label_from_data(self, data):
        return data[-self.SLICE_MARGIN:]
