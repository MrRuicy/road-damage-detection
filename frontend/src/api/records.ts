import request from './request'

export interface RecordItem {
  id: number
  detection_type: string
  source_name: string
  model_name: string
  conf_threshold: number
  iou_threshold: number
  total_damages: number
  class_counts: Record<string, number>
  severity: string
  severity_score: number
  result_image_path?: string
  boxes: any[]
  processing_time: number
  created_at?: string
  extra?: Record<string, any>
}

export interface RecordListResponse {
  total: number
  page: number
  page_size: number
  items: RecordItem[]
}

export interface RecordQuery {
  page?: number
  page_size?: number
  detection_type?: string
  severity?: string
  start_date?: string
  end_date?: string
}

// 历史记录列表（分页 + 筛选）
export const listRecords = (params: RecordQuery) => {
  return request.get<any, RecordListResponse>('/api/records', { params })
}

// 单条记录详情
export const getRecord = (id: number) => {
  return request.get<any, RecordItem>(`/api/records/${id}`)
}

// 删除记录
export const deleteRecord = (id: number) => {
  return request.delete(`/api/records/${id}`)
}
