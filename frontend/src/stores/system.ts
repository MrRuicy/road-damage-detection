import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { SystemStatus } from '../api/detection'
import { getSystemStatus, getAvailableModels, switchModel } from '../api/detection'

export const useSystemStore = defineStore('system', () => {
  const systemStatus = ref<SystemStatus | null>(null)
  const availableModels = ref<string[]>([])
  const currentModel = ref<string>('')
  const loading = ref(false)

  // 获取系统状态
  const fetchSystemStatus = async () => {
    try {
      loading.value = true
      systemStatus.value = await getSystemStatus()
    } catch (error) {
      console.error('获取系统状态失败:', error)
    } finally {
      loading.value = false
    }
  }

  // 获取可用模型
  const fetchAvailableModels = async () => {
    try {
      const data = await getAvailableModels()
      availableModels.value = data.models
      currentModel.value = data.current
    } catch (error) {
      console.error('获取模型列表失败:', error)
    }
  }

  // 切换模型
  const changeModel = async (modelName: string) => {
    try {
      await switchModel(modelName)
      currentModel.value = modelName
      return true
    } catch (error) {
      console.error('切换模型失败:', error)
      return false
    }
  }

  return {
    systemStatus,
    availableModels,
    currentModel,
    loading,
    fetchSystemStatus,
    fetchAvailableModels,
    changeModel
  }
})
