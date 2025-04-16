



class Trainer:
    def __init__(self, policy, env, buffer, batch_size, num_epochs):
        super().__init__()

        self.policy = policy
        self.env = env
        self.buffer = buffer
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def train(self):
        for epoch in range(self.num_epochs):

            # -------- 采集 batch_size 个轨迹样本（每一次产生一个样本），用buffer来存储 --------
            for _ in range(self.batch_size):
                traj = self.env.collect_traj()
                self.buffer.store_a_traj(traj)

            # -------- 取出本次采集的batch_size个样本 --------
            if not self.buffer.ready(batch_size=self.batch_size):   # 如果收集到的资料数目还没达到batch size，本轮不learn
                continue
            trajs = self.buffer.sample(batch_size=self.batch_size)

            # -------- 用样本来更新网络 --------
            self.policy.learn(trajs, self.buffer)

            # -------- 清空buffer，预备下次重新采集 --------
            self.buffer.clear()