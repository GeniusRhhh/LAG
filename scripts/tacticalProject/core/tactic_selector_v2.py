"""
战术选择器 V2 - 基于威胁值的智能选择
"""
import logging


class TacticSelectorV2:
    """战术选择器 - 基于威胁值"""
    
    def __init__(self, threat_evaluator):
        self.threat_evaluator = threat_evaluator
    
    def select_tactic_from_candidates(self, candidates, my_aircraft, enemy_aircraft, env):
        """
        从候选战术中选择最佳战术
        
        基于文档中的决策根据：
        - 拖曳射击：总威胁度差异大 (一机>0.8, 另一机<0.5)
        - 钳形攻势：角度威胁值大 (>0.8)
        - 上下夹击：高度威胁值大 (>0.8)
        - 前后攻击：威胁度和接近且较大
        - 并排射击：威胁度和接近且较小
        
        Args:
            candidates: 候选战术列表
            my_aircraft: 我方飞机列表 [长机, 僚机]
            enemy_aircraft: 敌方飞机列表
            env: 环境
        
        Returns:
            str: 选定的战术名称
        """
        try:
            # 计算威胁值
            lead_threats = []
            wingman_threats = []
            angle_threats = []
            altitude_threats = []
            
            # 我方长机和僚机
            my_lead = my_aircraft[0] if len(my_aircraft) > 0 else None
            my_wingman = my_aircraft[1] if len(my_aircraft) > 1 else None
            
            # 计算威胁值
            for enemy in enemy_aircraft:
                if enemy and enemy.is_alive:
                    if my_lead and my_lead.is_alive:
                        lead_threat = self.threat_evaluator.calculate_total_threat(my_lead, enemy, env)
                        lead_threats.append(lead_threat)
                        
                        angle_threat = self.threat_evaluator.calculate_angle_threat(my_lead, enemy)
                        angle_threats.append(angle_threat)
                        
                        altitude_threat = self.threat_evaluator.calculate_altitude_threat(my_lead, enemy)
                        altitude_threats.append(altitude_threat)
                    
                    if my_wingman and my_wingman.is_alive:
                        wingman_threat = self.threat_evaluator.calculate_total_threat(my_wingman, enemy, env)
                        wingman_threats.append(wingman_threat)
            
            # 计算平均威胁值
            avg_lead_threat = max(lead_threats) if lead_threats else 0.5
            avg_wingman_threat = max(wingman_threats) if wingman_threats else 0.5
            max_angle_threat = max(angle_threats) if angle_threats else 0.5
            max_altitude_threat = max(altitude_threats) if altitude_threats else 0.5
            total_threat_sum = avg_lead_threat + avg_wingman_threat
            
            # 战术选择逻辑
            scores = {}
            
            for tactic in candidates:
                score = 0.0
                
                if tactic == 'DRAG_SHOOT':
                    # 拖曳射击：威胁度差异大
                    threat_diff = abs(avg_lead_threat - avg_wingman_threat)
                    if (avg_lead_threat > 0.8 and avg_wingman_threat < 0.5) or \
                       (avg_wingman_threat > 0.8 and avg_lead_threat < 0.5):
                        score = 1.0
                    else:
                        score = threat_diff
                
                elif tactic == 'PINCER_ATTACK':
                    # 钳形攻势：角度威胁值大
                    if max_angle_threat > 0.8:
                        score = 1.0
                    else:
                        score = max_angle_threat
                
                elif tactic == 'HIGH_LOW_ATTACK':
                    # 上下夹击：高度威胁值大
                    if max_altitude_threat > 0.8:
                        score = 1.0
                    else:
                        score = max_altitude_threat
                
                elif tactic == 'SEQUENTIAL_ATTACK':
                    # 前后攻击：威胁度和接近且较大
                    threat_close = abs(avg_lead_threat - avg_wingman_threat) < 0.2
                    threat_high = total_threat_sum > 1.2
                    if threat_close and threat_high:
                        score = 1.0
                    else:
                        score = 0.5 if threat_close else 0.3
                
                elif tactic == 'SIDE_BY_SIDE':
                    # 并排射击：威胁度和接近且较小
                    threat_close = abs(avg_lead_threat - avg_wingman_threat) < 0.2
                    threat_low = total_threat_sum < 0.8
                    if threat_close and threat_low:
                        score = 1.0
                    else:
                        score = 0.5 if threat_close else 0.3
                
                scores[tactic] = score
            
            # 选择得分最高的战术
            if scores:
                selected_tactic = max(scores, key=scores.get)
                logging.info(f"战术选择: {selected_tactic} (得分: {scores[selected_tactic]:.2f})")
                logging.info(f"威胁值 - 长机: {avg_lead_threat:.2f}, 僚机: {avg_wingman_threat:.2f}, "
                           f"角度: {max_angle_threat:.2f}, 高度: {max_altitude_threat:.2f}")
                return selected_tactic
            else:
                return 'SIDE_BY_SIDE'
                
        except Exception as e:
            logging.error(f"战术选择错误: {e}")
            return 'SIDE_BY_SIDE'
