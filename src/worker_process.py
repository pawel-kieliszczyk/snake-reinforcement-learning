import mpi_helper
from learning_environment import LearningEnvironment
from a2c import Model, A2CAgent


class Worker(object):
    def __init__(self, id):
        self.id = id

    def run(self, epochs, batch_size):
        model = Model()
        agent = A2CAgent(model)
        learning_environment = LearningEnvironment()

        agent.initialize_model(learning_environment)
        variables = model.get_variables()

        for _ in range(epochs):
            variables = self._receive_variables_from_master(variables)
            model.set_variables(variables)

            observations, acts_and_advs, returns = agent.generate_experience_batch(learning_environment, batch_size)
            self._send_experience_to_master(observations, acts_and_advs, returns)

    def _receive_variables_from_master(self, variables):
        for v in variables:
            mpi_helper.broadcast(v)
        return variables

    def _send_experience_to_master(self, observations, acts_and_advs, returns):
        mpi_helper.send(observations, dest=0)
        mpi_helper.send(acts_and_advs, dest=0)
        mpi_helper.send(returns, dest=0)
