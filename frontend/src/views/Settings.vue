<script setup lang="ts">
import { onMounted } from 'vue'
import { useSystemStore } from '../stores/system'
import { useThemeStore } from '../stores/theme'
import { ElMessage } from 'element-plus'

const systemStore = useSystemStore()
const themeStore = useThemeStore()

onMounted(() => {
  systemStore.fetchAvailableModels()
})

const handleModelChange = async (modelName: string) => {
  const success = await systemStore.changeModel(modelName)
  if (success) {
    ElMessage.success(`已切换到模型: ${modelName}`)
  }
}
</script>

<template>
  <div class="page">
    <div class="page-header">
      <h1>系统设置</h1>
      <p>配置系统参数</p>
    </div>

    <el-card shadow="hover" class="settings-card">
      <template #header>
        <span>模型设置</span>
      </template>
      <el-form label-width="120px">
        <el-form-item label="默认模型">
          <el-select
            :model-value="systemStore.currentModel"
            @change="handleModelChange"
            style="width: 300px"
          >
            <el-option
              v-for="model in systemStore.availableModels"
              :key="model"
              :label="model"
              :value="model"
            />
          </el-select>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card shadow="hover" class="settings-card" style="margin-top: 24px">
      <template #header>
        <span>外观设置</span>
      </template>
      <el-form label-width="120px">
        <el-form-item label="深色模式">
          <el-switch
            :model-value="themeStore.isDark"
            @change="(val: any) => themeStore.setDark(Boolean(val))"
            inline-prompt
            active-text="开"
            inactive-text="关"
          />
        </el-form-item>
      </el-form>
    </el-card>

    <el-card shadow="hover" class="settings-card" style="margin-top: 24px">
      <template #header>
        <span>关于系统</span>
      </template>
      <el-descriptions :column="1" border>
        <el-descriptions-item label="系统名称">道路病害检测系统</el-descriptions-item>
        <el-descriptions-item label="版本">v2.0.0</el-descriptions-item>
        <el-descriptions-item label="技术栈">Vue 3 + FastAPI + YOLO11</el-descriptions-item>
      </el-descriptions>
    </el-card>
  </div>
</template>

<style scoped>
.page {
  padding: 0;
}

.page-header {
  margin-bottom: 24px;
}

.page-header h1 {
  margin: 0 0 8px;
  font-size: 28px;
}

.page-header p {
  margin: 0;
}

.settings-card {
  border-radius: 12px;
}
</style>
