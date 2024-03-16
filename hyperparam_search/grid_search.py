import logging
class GridSearch:
    def __init__(self, params_dict):
        self.params_dict = params_dict
        self.indices = {}
        for name in self.params_dict.keys():
            self.indices[name] = [0, len(self.params_dict[name])]
    
    def grid_search(self):
        logger = logging.getLogger()
        # logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.DEBUG)

        params = {}
        flag = True
        for i in self.indices.keys():
            params[i] = self.params_dict[i][self.indices[i][0]]
            if flag and self.indices[i][0] < self.indices[i][1]-1:
                flag = False
                self.indices[i][0] += 1
        if flag:
            logger.info('Complete grid search')
            return None
        else:
            logger.info(f'new param {params}')
            return params


# if __name__ == '__main__':
#     gs = GridSearch({'drop': [0.3,0.5,0.7], 'bn': [True, False]})
#     for i in range(15):
#         a = gs.grid_search()
#         print(a)