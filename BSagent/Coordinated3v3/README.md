# Triad 3v3

Triad 是本仓库的独立比赛 Agent，ID 为 `com.bsfootballgame.triad3v3`。它沿用 THU-Ichin 的 Coordinated3v3 和 Booster 官方行为树。HuangJianshi 的 Kinger 提供了单一主罚手、定位球方向保护和后场分工的策略参考。两套参考项目的原始版本可从各自的分支取得。

- **开球与定位球**：根据裁判数据锁定一名主罚手，当前球员不可用时才换人。中圈开球时其余球员原地等候。间接任意球、界外球、球门球和中圈开球的第一脚优先拨向场内安全线路。
- **防守**：按有效位姿和裁判状态分配门将；门将不可用时由离球门最近的可用球员补位。只剩门将时，球在己方危险区域内才出击。滚地球预计能到达球门时，门将提前往球路上移动。
- **进攻**：保留滚地球拦截、接应和选角射门；每帧重用出球目标，传球时排除不可用球员，并沿所选踢球方向从球后接近。
- **裁判保护**：无有效裁判数据、暂停、SET 和比赛结束时全队停车。在暂停或受罚时不发起起身动作。策略沿用官方示例的 ROS 输入和机器人控制接口。

## Booster Studio

在 Studio 中打开 `BSagent/Coordinated3v3`，选择 **Virtual Robot** 和 **football3v3**，执行 **Activate, build, deploy, and run agent**。构建完成后使用该版本生成的 `.agent` 文件。仓库中的 Kinger 包是另一个 Agent 的产物。

用同一个 Booster AI 跑红蓝换边对照，保存版本、比分、犯规数和回放。回放优先检查暂停或 SET、双方开球及定位球、门将受罚后的补位、以及近门滚地球。

## 离线验证

以下命令在本目录运行。行为树测试使用合成的裁判、位姿和机器人状态，覆盖两侧球队和比赛状态变化。云端物理仿真、资格赛结果和胜率需要单独验证。

```sh
python -m pip install py_trees==2.4.0
python -m unittest discover -s tests -v
```

参照 [Booster 官方示例](https://github.com/BoosterRobotics/booster_champion_example/tree/178fd9c)、[官方简化示例](https://github.com/BoosterRobotics/sim-3v3-simple-framework)、[开发协议](docs/developer_protocol.md)和[赛场规则](https://booster.feishu.cn/wiki/V5oAwY9QjierOYk6PkncdilUnhe)。
