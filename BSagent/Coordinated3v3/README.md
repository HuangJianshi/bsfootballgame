# Coordinated 3v3

这是基于 [Booster 官方行为树示例](https://github.com/BoosterRobotics/booster_champion_example/tree/178fd9c) 的 3v3 足球策略。

## 本地测试

1. 启动 Docker Desktop。在 Booster Studio 中打开本目录 `BSagent/Coordinated3v3`。
2. 选择 **Virtual Robot** 和 **football3v3** 场景。运行 **Activate, build, deploy, and run agent**。
3. 在仿真窗口点击 **Start**。可以先选内置 **Booster AI** 作为对手，在 Studio 的 **Output → Agent Runtime** 查看球员动作。

## 策略入口

- `src/play/coordinated_support.py` 包含外场球员角色交接和后场防守分工。
- `src/play/ball_motion.py`、`src/play/rolling_intercept.py` 包含滚地球预测和截球规划。
- `src/tactics/targeting/attack.py` 包含射门选角、近门直接射门和封堵下的出球选择。
- `src/tactics/motion.py` 包含近球侧移和出球预对准。

建议先验证门前交接、两名外场球员换位及中路被封时的出球方向。提交测试结果时附上对手、比分、比赛时间和对应的 Agent Runtime 日志片段。

场景数据接口见 [开发协议](docs/developer_protocol.md)，行为树结构见 [行为树说明](docs/bt_structure.md)。
