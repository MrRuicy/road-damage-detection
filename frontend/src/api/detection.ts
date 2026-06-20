import request from './request'

export interface DetectionBox {
  x1: number
  y1: number
  x2: number
  y2: number
  confidence: number
  class_id: number
  class_name: string
}

export interface DetectionResult {
  image_name: string
  boxes: DetectionBox[]
  class_counts: Record<string, number>
  total_damages: number
  severity: string
  severity_score: number
  processing_time: number
  result_image?: string
  record_id?: number
  is_duplicate?: boolean
}

export interface BatchDetectionResult {
  total_images: number
  total_damages: number
  overall_severity: string
  overall_severity_score: number
  class_counts: Record<string, number>
  results: DetectionResult[]
}

export interface SystemStatus {
  cpu_usage: number
  memory_usage: number
  gpu_usage: number
  gpu_memory?: number
  current_model: string
  available_models: string[]
}

// 图像检测
export const detectImage = (formData: FormData) => {
  return request.post<any, DetectionResult>('/api/detection/image', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
}

// 批量检测
export const detectBatch = (formData: FormData) => {
  return request.post<any, BatchDetectionResult>('/api/detection/batch', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
}

// 获取系统状态
export const getSystemStatus = () => {
  return request.get<any, SystemStatus>('/api/system/status')
}

// 切换模型
export const switchModel = (modelName: string) => {
  return request.post(`/api/system/model/switch?model_name=${encodeURIComponent(modelName)}`)
}

// 获取可用模型
export const getAvailableModels = () => {
  return request.get<any, { models: string[], current: string }>('/api/system/models')
}
