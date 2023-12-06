def get_text_from_data(data, slice_margin):
    return data[1:-slice_margin-2]


def get_label_from_data(data, slice_margin):
    return data[-slice_margin:]


class Review:
    text = ""
    label = ""

    def __init__(self, data, slice_margin=8):
        self.text = get_text_from_data(data, slice_margin)
        self.label = get_label_from_data(data, slice_margin)
