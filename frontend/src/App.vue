<script setup lang="ts">
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useThemeStore } from './stores/theme'

const router = useRouter()
const route = useRoute()
const themeStore = useThemeStore()

// 高亮跟随当前路由，无论从菜单还是快捷操作跳转都同步
const activeMenu = computed(() => route.path)

const handleMenuSelect = (index: string) => {
  router.push(index)
}
</script>

<template>
  <el-container class="app-container">
    <el-aside width="240px" class="app-aside">
      <div class="logo">
        <el-icon :size="32"><Position /></el-icon>
        <span class="logo-text">道路病害检测</span>
      </div>
      <el-menu
        :default-active="activeMenu"
        class="app-menu"
        @select="handleMenuSelect"
      >
        <el-menu-item index="/dashboard">
          <el-icon><Odometer /></el-icon>
          <span>仪表板</span>
        </el-menu-item>
        <el-menu-item index="/detection">
          <el-icon><Picture /></el-icon>
          <span>图像检测</span>
        </el-menu-item>
        <el-menu-item index="/video">
          <el-icon><VideoCamera /></el-icon>
          <span>视频检测</span>
        </el-menu-item>
        <el-menu-item index="/realtime">
          <el-icon><View /></el-icon>
          <span>实时检测</span>
        </el-menu-item>
        <el-menu-item index="/history">
          <el-icon><Clock /></el-icon>
          <span>检测历史</span>
        </el-menu-item>
        <el-menu-item index="/settings">
          <el-icon><Setting /></el-icon>
          <span>系统设置</span>
        </el-menu-item>
      </el-menu>

      <div class="aside-footer">
        <el-button text class="theme-toggle" @click="themeStore.toggle">
          <el-icon :size="18">
            <Moon v-if="!themeStore.isDark" />
            <Sunny v-else />
          </el-icon>
          <span>{{ themeStore.isDark ? '浅色模式' : '深色模式' }}</span>
        </el-button>
      </div>
    </el-aside>

    <el-container class="main-container">
      <el-main class="app-main">
        <router-view v-slot="{ Component }">
          <transition name="fade" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<style scoped>
.app-container {
  height: 100vh;
}

.app-aside {
  background: #001529;
  box-shadow: 2px 0 8px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
}

.logo {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 64px;
  color: #fff;
  font-size: 18px;
  font-weight: bold;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  gap: 12px;
  flex-shrink: 0;
}

.logo-text {
  background: linear-gradient(120deg, #4facfe 0%, #00f2fe 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.app-menu {
  border: none;
  background: #001529;
  flex: 1;
}

.app-menu :deep(.el-menu-item) {
  color: rgba(255, 255, 255, 0.65);
}

.app-menu :deep(.el-menu-item:hover) {
  color: #fff;
  background: rgba(255, 255, 255, 0.08);
}

.app-menu :deep(.el-menu-item.is-active) {
  color: #fff;
  background: linear-gradient(120deg, #4facfe 0%, #00f2fe 100%);
}

.aside-footer {
  padding: 16px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  flex-shrink: 0;
}

.theme-toggle {
  width: 100%;
  color: rgba(255, 255, 255, 0.65);
  justify-content: flex-start;
  gap: 8px;
}

.theme-toggle:hover {
  color: #fff;
  background: rgba(255, 255, 255, 0.08);
}

.main-container {
  background: var(--el-bg-color-page);
}

.app-main {
  padding: 24px;
  overflow-y: auto;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
