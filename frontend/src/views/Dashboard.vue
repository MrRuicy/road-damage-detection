<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useSystemStore } from '../stores/system'
import {
  getOverview,
  getClassDistribution,
  getSeverityDistribution,
  getTrend,
  type StatsOverview,
  type TrendPoint
} from '../api/stats'

const systemStore = useSystemStore()

const overview = ref<StatsOverview>({
  total_detections: 0,
  today_detections: 0,
  week_detections: 0,
  month_detections: 0,
  total_damages: 0,
  avg_severity_score: 0
})
const trend = ref<TrendPoint[]>([])
const classDist = ref<Record<string, number>>({})
const severityDist = ref<Record<string, number>>({})

const SEVERITY_COLORS: Record<string, string> = {
  无病害: '#67c23a',
  轻微: '#85ce61',
  中等: '#e6a23c',
  严重: '#f56c6c',
  危险: '#c0392b'
}

// 趋势折线图
const trendOption = computed(() => ({
  tooltip: { trigger: 'axis' },
  legend: { data: ['检测次数', '病害数'], bottom: 0 },
  grid: { left: 40, right: 20, top: 30, bottom: 40 },
  xAxis: {
    type: 'category',
    data: trend.value.map(t => t.date.slice(5)),
    boundaryGap: false
  },
  yAxis: { type: 'value' },
  series: [
    {
      name: '检测次数',
      type: 'line',
      smooth: true,
      data: trend.value.map(t => t.detections),
      areaStyle: { opacity: 0.15 },
      itemStyle: { color: '#409eff' }
    },
    {
      name: '病害数',
      type: 'line',
      smooth: true,
      data: trend.value.map(t => t.damages),
      areaStyle: { opacity: 0.15 },
      itemStyle: { color: '#f56c6c' }
    }
  ]
}))

// 病害类型分布饼图
const classOption = computed(() => ({
  tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
  legend: { bottom: 0 },
  series: [
    {
      name: '病害类型',
      type: 'pie',
      radius: ['40%', '70%'],
      avoidLabelOverlap: true,
      itemStyle: { borderRadius: 6, borderColor: '#fff', borderWidth: 2 },
      label: { show: false, position: 'center' },
      emphasis: { label: { show: true, fontSize: 18, fontWeight: 'bold' } },
      data: Object.entries(classDist.value).map(([name, value]) => ({ name, value }))
    }
  ]
}))

// 严重度分布柱状图
const severityOption = computed(() => {
  const order = ['无病害', '轻微', '中等', '严重', '危险']
  const entries = order.filter(k => k in severityDist.value)
  return {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 40, right: 20, top: 20, bottom: 30 },
    xAxis: { type: 'category', data: entries },
    yAxis: { type: 'value' },
    series: [
      {
        type: 'bar',
        data: entries.map(k => ({
          value: severityDist.value[k],
          itemStyle: { color: SEVERITY_COLORS[k] || '#909399' }
        })),
        barWidth: '50%',
        itemStyle: { borderRadius: [6, 6, 0, 0] }
      }
    ]
  }
})

const hasClassData = computed(() => Object.keys(classDist.value).length > 0)
const hasSeverityData = computed(() => Object.keys(severityDist.value).length > 0)

const loadStats = async () => {
  const [ov, tr, cd, sd] = await Promise.all([
    getOverview(),
    getTrend(7),
    getClassDistribution(),
    getSeverityDistribution()
  ])
  overview.value = ov
  trend.value = tr
  classDist.value = cd
  severityDist.value = sd
}

onMounted(() => {
  systemStore.fetchSystemStatus()
  systemStore.fetchAvailableModels()
  loadStats()
})
</script>
<template>
  <div class="dashboard">
    <div class="page-header">
      <h1>仪表板</h1>
      <p>道路病害检测系统数据总览</p>
    </div>

    <!-- 统计卡片 -->
    <el-row :gutter="24" class="stats-row">
      <el-col :xs="12" :sm="12" :lg="6">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-content">
            <el-icon class="stat-icon" color="#409eff"><Histogram /></el-icon>
            <div class="stat-info">
              <div class="stat-label">累计检测</div>
              <div class="stat-value">{{ overview.total_detections }}</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :xs="12" :sm="12" :lg="6">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-content">
            <el-icon class="stat-icon" color="#67c23a"><Calendar /></el-icon>
            <div class="stat-info">
              <div class="stat-label">今日检测</div>
              <div class="stat-value">{{ overview.today_detections }}</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :xs="12" :sm="12" :lg="6">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-content">
            <el-icon class="stat-icon" color="#e6a23c"><Warning /></el-icon>
            <div class="stat-info">
              <div class="stat-label">病害总数</div>
              <div class="stat-value">{{ overview.total_damages }}</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :xs="12" :sm="12" :lg="6">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-content">
            <el-icon class="stat-icon" color="#f56c6c"><TrendCharts /></el-icon>
            <div class="stat-info">
              <div class="stat-label">平均严重指数</div>
              <div class="stat-value">{{ overview.avg_severity_score }}</div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 趋势图 -->
    <el-card shadow="hover" class="chart-card">
      <template #header><span>近 7 天检测趋势</span></template>
      <v-chart class="chart" :option="trendOption" autoresize />
    </el-card>

    <!-- 分布图 -->
    <el-row :gutter="24" class="chart-row">
      <el-col :xs="24" :lg="12">
        <el-card shadow="hover" class="chart-card">
          <template #header><span>病害类型分布</span></template>
          <v-chart v-if="hasClassData" class="chart" :option="classOption" autoresize />
          <el-empty v-else description="暂无数据" :image-size="80" />
        </el-card>
      </el-col>
      <el-col :xs="24" :lg="12">
        <el-card shadow="hover" class="chart-card">
          <template #header><span>严重度分布</span></template>
          <v-chart v-if="hasSeverityData" class="chart" :option="severityOption" autoresize />
          <el-empty v-else description="暂无数据" :image-size="80" />
        </el-card>
      </el-col>
    </el-row>

    <!-- 系统状态 + 快捷操作 -->
    <el-row :gutter="24" class="chart-row">
      <el-col :xs="24" :lg="12">
        <el-card shadow="hover" class="status-card">
          <template #header>
            <div class="card-header">
              <span>系统状态</span>
              <el-button text @click="systemStore.fetchSystemStatus">
                <el-icon><Refresh /></el-icon>
              </el-button>
            </div>
          </template>
          <div v-if="systemStore.systemStatus" class="status-content">
            <div class="status-item">
              <span class="status-label">CPU 使用率</span>
              <el-progress :percentage="Math.round(systemStore.systemStatus.cpu_usage)" />
            </div>
            <div class="status-item">
              <span class="status-label">内存使用率</span>
              <el-progress :percentage="Math.round(systemStore.systemStatus.memory_usage)" :color="'#67c23a'" />
            </div>
            <div class="status-item">
              <span class="status-label">GPU 使用率</span>
              <el-progress :percentage="Math.round(systemStore.systemStatus.gpu_usage)" :color="'#e6a23c'" />
            </div>
            <div class="status-item model-line">
              当前模型：<el-tag size="small">{{ systemStore.currentModel || '加载中' }}</el-tag>
            </div>
          </div>
          <el-empty v-else description="加载中..." :image-size="60" />
        </el-card>
      </el-col>
      <el-col :xs="24" :lg="12">
        <el-card shadow="hover" class="status-card">
          <template #header><span>快捷操作</span></template>
          <div class="quick-grid">
            <el-button type="primary" size="large" @click="$router.push('/detection')">
              <el-icon><Picture /></el-icon>图像检测
            </el-button>
            <el-button type="success" size="large" @click="$router.push('/video')">
              <el-icon><VideoCamera /></el-icon>视频检测
            </el-button>
            <el-button type="warning" size="large" @click="$router.push('/realtime')">
              <el-icon><View /></el-icon>实时检测
            </el-button>
            <el-button type="info" size="large" @click="$router.push('/history')">
              <el-icon><Clock /></el-icon>检测历史
            </el-button>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.dashboard {
  padding: 0;
}

.page-header {
  margin-bottom: 24px;
}

.page-header h1 {
  margin: 0 0 8px;
  font-size: 28px;
  color: #303133;
}

.page-header p {
  margin: 0;
  color: #909399;
}

.stats-row {
  margin-bottom: 24px;
}

.stat-card {
  border: none;
  border-radius: 12px;
  transition: all 0.3s ease;
  margin-bottom: 16px;
}

.stat-card:hover {
  transform: translateY(-4px);
}

.stat-content {
  display: flex;
  align-items: center;
  gap: 16px;
}

.stat-icon {
  font-size: 44px;
  opacity: 0.85;
}

.stat-info {
  flex: 1;
  min-width: 0;
}

.stat-label {
  font-size: 14px;
  color: #909399;
  margin-bottom: 6px;
}

.stat-value {
  font-size: 28px;
  font-weight: bold;
  color: #303133;
}

.chart-card {
  border-radius: 12px;
  margin-bottom: 24px;
}

.chart {
  height: 320px;
  width: 100%;
}

.chart-row {
  margin-bottom: 0;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.status-card {
  border-radius: 12px;
  margin-bottom: 24px;
  height: calc(100% - 24px);
}

.status-content {
  padding: 8px 0;
}

.status-item {
  margin-bottom: 18px;
}

.status-label {
  display: block;
  margin-bottom: 8px;
  color: #606266;
  font-size: 14px;
}

.model-line {
  color: #606266;
  font-size: 14px;
}

.quick-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.quick-grid .el-button {
  width: 100%;
  margin: 0;
}
</style>
