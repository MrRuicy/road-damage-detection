import request from './request'

// 触发浏览器下载 blob
const downloadBlob = (blob: Blob, filename: string) => {
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  window.URL.revokeObjectURL(url)
}

// 导出单条记录的 PDF 报告
export const exportPdf = async (recordId: number) => {
  const blob = await request.get<any, Blob>(`/api/export/pdf/${recordId}`, {
    responseType: 'blob'
  })
  downloadBlob(blob, `检测报告_${recordId}.pdf`)
}

// 导出记录为 Excel（带筛选条件）
export const exportExcel = async (params: Record<string, any> = {}) => {
  const blob = await request.get<any, Blob>('/api/export/records.xlsx', {
    params,
    responseType: 'blob'
  })
  const ts = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '')
  downloadBlob(blob, `检测记录_${ts}.xlsx`)
}
