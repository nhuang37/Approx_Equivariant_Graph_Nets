import numpy as np
import torch

from lib import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type
        #add
        self._params_copy_dict = {}

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=device))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]

    def copy_weights(self, shape, idx=0):
        #initialize weights for different orbits the same way 
        #append id to differentiate different weight copies
        nn_param = torch.nn.Parameter(self._params_dict[shape])        
        self._params_copy_dict[(shape, idx)] = nn_param
        self._rnn_network.register_parameter('{}_weight_copy_{}_{}'.format(self._type, str(shape), str(idx)),
                                             nn_param)
        return self._params_copy_dict[(shape, idx)]


class DCGRUCell(torch.nn.Module):
    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True, aut_flag=False, orbits=None, idx_swap_back=None):
        """
        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        :param aut_flag: whether to use aut(G)-equivariant layer (i.e. local weight sharing)
        :param orbits: a list of sublist, where each sublist is the set of nodes that share weights
        :param idx_swap_back: invert back the orbit partition to the original node ID ordering in the graph
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru
        self.aut_flag = aut_flag
        self.orbits = orbits
        self.idx_swap_back = idx_swap_back

        supports = []
        if filter_type == "laplacian":
            supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
            supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(utils.calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))

        self._fc_params = LayerParams(self, 'fc')
        self._gconv_params = LayerParams(self, 'gconv')

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)
        return L

    def forward(self, inputs, hx):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)
        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self._gconv
        else:
            fn = self._fc
        #default to MPNN - shared weights for learning gating vectors
        value = torch.sigmoid(fn(inputs, hx, output_size, bias_start=1.0, aut_flag=False))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(inputs, r * hx, self._num_units, aut_flag=self.aut_flag) #here optionally change to local-weight sharing
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0, aut_flag=False):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.matmul(inputs_and_state, weights)
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value

    def _gconv(self, inputs, state, output_size, bias_start=0.0, aut_flag=True):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0) #(1, num_nodes, input_size * batch_size)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)

        #MPNN part remains the same, only modify the linear transformation below
        if aut_flag:
            num_orbits = len(self.orbits)
            weights_list = [self._gconv_params.get_weights((input_size * num_matrices, output_size))]
            for i in range(num_orbits - 1):
                weights_list.append(self._gconv_params.copy_weights(shape=(input_size * num_matrices, output_size), idx=i))
            biases = self._gconv_params.get_biases(output_size * num_orbits, bias_start)

            out = []
            for i, orbit in enumerate(self.orbits):
                x_sub = x[:, orbit, :, :]
                x_sub = torch.reshape(x_sub, shape=[batch_size * len(orbit), input_size * num_matrices])
                out_sub = torch.matmul(x_sub, weights_list[i])
                out_sub += biases[output_size * i : output_size * (i+1) ]
                out_sub = torch.reshape(out_sub, shape=[batch_size, len(orbit), output_size])
                out.append(out_sub) #(batch_size,  orbit_size, output_size)
            out = torch.cat(out, dim=1) #(batch_size, self._num_nodes, output_size)
            #out = torch.reshape(out, shape=[batch_size, self._num_nodes, output_size]) #decompose back the nodes dimension
            out = out[:, self.idx_swap_back, :] #re-ordering the nodes
            return torch.reshape(out, [batch_size, self._num_nodes * output_size])

        else: #original
            x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])
            weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
            x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

            biases = self._gconv_params.get_biases(output_size, bias_start)
            x += biases
            # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
            return torch.reshape(x, [batch_size, self._num_nodes * output_size])