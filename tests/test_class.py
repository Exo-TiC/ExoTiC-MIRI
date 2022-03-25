class Extract1dStep(object):
    spec = """
    extract_algo = option("box", "optimal", default="box")  # extraction algorithm
    """

    reference_file_types = ['extract1d', 'apcorr', 'wavemap', 'spectrace', 'specprofile']

    def process(self, input):
        return


a = Extract1dStep()
print(a.spec)
print(a.reference_file_types)
