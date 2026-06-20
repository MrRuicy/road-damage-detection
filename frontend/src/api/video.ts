import request from './request'

export interface VideoTaskResult {
  video_name: string
  result_video_url: string
  total_frames: number
  processed_frames: number
  fps: number
  total_damages: number
  class_counts: Record<string, number>
  severity: string
  severity_score: number
  processing_time: number
}

export interface VideoTaskStatus {
  id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  message: string
  total_frames: number
  processed_frames: number
  result?: VideoTaskResult
  error?: string
}

// 上传视频，启动后台检测任务
export const detectVideo = (formData: FormData) => {
  return request.post<any, { task_id: string; message: string }>(
    '/api/detection/video',
    formData,
    { headers: { 'Content-Type': 'multipart/form-data' } }
  )
}

// 查询视频任务进度
export const getVideoTask = (taskId: string) => {
  return request.get<any, VideoTaskStatus>(`/api/detection/video/task/${taskId}`)
}
