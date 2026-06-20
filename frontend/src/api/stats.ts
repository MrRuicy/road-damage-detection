import request from './request'

export interface StatsOverview {
  total_detections: number
  today_detections: number
  week_detections: number
  month_detections: number
  total_damages: number
  avg_severity_score: number
}

export interface TrendPoint {
  date: string
  detections: number
  damages: number
}

// 总览统计
export const getOverview = () => {
  return request.get<any, StatsOverview>('/api/stats/overview')
}

// 病害类型分布
export const getClassDistribution = () => {
  return request.get<any, Record<string, number>>('/api/stats/class-distribution')
}

// 严重度等级分布
export const getSeverityDistribution = () => {
  return request.get<any, Record<string, number>>('/api/stats/severity-distribution')
}

// 检测趋势
export const getTrend = (days = 7) => {
  return request.get<any, TrendPoint[]>('/api/stats/trend', { params: { days } })
}
