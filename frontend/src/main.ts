import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import 'element-plus/theme-chalk/dark/css-vars.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import VChart from 'vue-echarts'
import router from './router'
import App from './App.vue'
import { useThemeStore } from './stores/theme'
import './plugins/echarts'
import './style.css'

const app = createApp(App)
const pinia = createPinia()

// 注册所有 Element Plus 图标
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

// 全局注册 ECharts 组件
app.component('VChart', VChart)

app.use(pinia)
app.use(router)
app.use(ElementPlus)

// 初始化主题（读取 localStorage）
useThemeStore().init()

app.mount('#app')
