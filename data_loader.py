from random import shuffle


class DataIterator(object):
    def __init__(self, data, batch_size):
        self.data = data
        self.idx = 0
        self.limit = len(data)
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def next(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.idx + batch_size < self.limit:
            output = self.data[self.idx: self.idx + batch_size]
            self.idx += batch_size
        else:
            shuffle(self.data)
            output = self.data[0: batch_size]
            self.idx = batch_size
        return output


class DataLoader(object):
    def __init__(self, batch_size):
        '''this is init'''
        self.datasets = {}
        self.batch_size = batch_size

    def new_dataset(self, name, data, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if not isinstance(name, str):
            raise TypeError()
        elif name in self.datasets:
            raise NameError(name +
                            ' already exists as a dataset. Give new name')
        else:
            self.datasets[name] = DataIterator(data, batch_size)

    def next_batch(self, name, batch_size=None):
        return self.datasets[name].next(batch_size)
