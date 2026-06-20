import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useThemeStore = defineStore('theme', () => {
  const isDark = ref(false)

  // 应用主题到 <html>，Element Plus 通过 .dark 类切换暗色变量
  const applyTheme = () => {
    const html = document.documentElement
    if (isDark.value) {
      html.classList.add('dark')
    } else {
      html.classList.remove('dark')
    }
  }

  const toggle = () => {
    isDark.value = !isDark.value
    localStorage.setItem('theme', isDark.value ? 'dark' : 'light')
    applyTheme()
  }

  const setDark = (val: boolean) => {
    isDark.value = val
    localStorage.setItem('theme', val ? 'dark' : 'light')
    applyTheme()
  }

  // 初始化：读取本地存储
  const init = () => {
    const saved = localStorage.getItem('theme')
    isDark.value = saved === 'dark'
    applyTheme()
  }

  return { isDark, toggle, setDark, init }
})
