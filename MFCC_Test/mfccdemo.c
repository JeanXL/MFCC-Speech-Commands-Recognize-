#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "wave.h"

#define M_2PI 6.2831853
#define ORIGN_WAV_LEN   16000
#define RESAMPLE_WAV_LEN    8000
#define frame_length_ms 32
#define frame_shift_ms  20
#define num_frames  50
#define frame_length 256
#define frame_step  160

int16_t buff[ORIGN_WAV_LEN];
int16_t wave_8k[RESAMPLE_WAV_LEN];
float emph_wave_8k[RESAMPLE_WAV_LEN];

float frames[num_frames][frame_length] = { 0.f };
float window_func[frame_length] = { 0.f };
float pow_frames[num_frames][frame_length / 2 + 1] = { 0.f };

float fft_x[frame_length] = { 0.f };
float fbank[40][frame_length / 2 + 1] = { 0.f };
float filter_banks[num_frames][40] = { 0.f };
float dct_buff[12][40] = { 0.f };
float mfcc[50][12] = { 0.f };

float* mfcc_data = mfcc;
float a_min = 0.f;
float a_max = 0.f;
uint16_t cnt = 0;

void rfft(float x[], int n)
{
    int i, j, k, m, i1, i2, i3, i4, n1, n2, n4;
    float a, e, cc, ss, xt, t1, t2;
    for (j = 1, i = 1; i < 16; i++)
    {
        m = i;
        j = 2 * j;
        if (j == n)
            break;
    }
    n1 = n - 1;
    for (j = 0, i = 0; i < n1; i++)
    {
        if (i < j)
        {
            xt = x[j];
            x[j] = x[i];
            x[i] = xt;
        }
        k = n / 2;
        while (k < (j + 1))
        {
            j = j - k;
            k = k / 2;
        }
        j = j + k;
    }
    for (i = 0; i < n; i += 2)
    {
        xt = x[i];
        x[i] = xt + x[i + 1];
        x[i + 1] = xt - x[i + 1];
    }
    n2 = 1;
    for (k = 2; k <= m; k++)
    {
        n4 = n2;
        n2 = 2 * n4;
        n1 = 2 * n2;
        e = 6.28318530718 / n1;
        for (i = 0; i < n; i += n1)
        {
            xt = x[i];
            x[i] = xt + x[i + n2];
            x[i + n2] = xt - x[i + n2];
            x[i + n2 + n4] = -x[i + n2 + n4];
            a = e;
            for (j = 1; j <= (n4 - 1); j++)
            {
                i1 = i + j;
                i2 = i - j + n2;
                i3 = i + j + n2;
                i4 = i - j + n1;
                cc = cos(a);
                ss = sin(a);
                a = a + e;
                t1 = cc * x[i3] + ss * x[i4];
                t2 = ss * x[i3] - cc * x[i4];
                x[i4] = x[i2] - t2;
                x[i3] = -x[i2] - t2;
                x[i2] = x[i1] - t1;
                x[i1] = x[i1] + t1;
            }
        }
    }
}

int main()
{
    uint16_t i,j,k;

    FILE* fp = NULL;

    Wav wav;
    RIFF_t riff;
    FMT_t fmt;
    Data_t data;

    errno_t err;
    err = fopen_s(&fp, "./wav_file/tetsno.wav", "rb");
    if (err) {
        printf("can't open audio file\n");
        exit(1);
    }

    fread(&wav, 1, sizeof(wav), fp);

    riff = wav.riff;
    fmt = wav.fmt;
    data = wav.data;

    printf("ChunkID \t%c%c%c%c\n", riff.ChunkID[0], riff.ChunkID[1], riff.ChunkID[2], riff.ChunkID[3]);
    printf("ChunkSize \t%d\n", riff.ChunkSize);
    printf("Format \t\t%c%c%c%c\n", riff.Format[0], riff.Format[1], riff.Format[2], riff.Format[3]);

    printf("\n");

    printf("Subchunk1ID \t%c%c%c%c\n", fmt.Subchunk1ID[0], fmt.Subchunk1ID[1], fmt.Subchunk1ID[2], fmt.Subchunk1ID[3]);
    printf("Subchunk1Size \t%d\n", fmt.Subchunk1Size);
    printf("AudioFormat \t%d\n", fmt.AudioFormat);
    printf("NumChannels \t%d\n", fmt.NumChannels);
    printf("SampleRate \t%d\n", fmt.SampleRate);
    printf("ByteRate \t%d\n", fmt.ByteRate);
    printf("BlockAlign \t%d\n", fmt.BlockAlign);
    printf("BitsPerSample \t%d\n", fmt.BitsPerSample);

    printf("\n");

    printf("blockID \t%c%c%c%c\n", data.Subchunk2ID[0], data.Subchunk2ID[1], data.Subchunk2ID[2], data.Subchunk2ID[3]);
    printf("blockSize \t%d\n", data.Subchunk2Size);

    printf("\n");

    printf("duration \t%d\n", data.Subchunk2Size / fmt.ByteRate);

    fread(&buff, 1, sizeof(buff), fp);  // 采样频率16K，时间为1s，总的点数为16K

    fclose(fp);

    /***************** MFCC **********************/
    // 1. 下采样，16K---8K
    for (i = 0; i < RESAMPLE_WAV_LEN; i++)
    {
        wave_8k[i] = buff[2 * i];
    }

    // 2. 预加重，实则为高通滤波，突出语音信号中的高频共振峰y(t)=x(t)−αx(t−1)
    emph_wave_8k[0] = wave_8k[0];
    for (i = 1; i < RESAMPLE_WAV_LEN; i++)
    {
        emph_wave_8k[i] = wave_8k[i] - 0.97 * wave_8k[i - 1];
    }

    // 3. 分帧
    for (i = 0; i < num_frames; i++) 
    {
        for (j = 0; j < frame_length; j++)
        {
            if (cnt >= 8000)
            {
                frames[i][j] = 0.0;
            }
            else
            {
                frames[i][j] = emph_wave_8k[cnt++];
            }
        }
        cnt = cnt - 96;
    }

    // 4.加窗
    for (int i = 0; i < frame_length; i++)
        window_func[i] = 0.54 - 0.46 * cos(M_2PI * ((float)i) / ((float)frame_length - 1));
    for (i = 0; i < num_frames; i++)
    {
        for (j = 0; j < frame_length; j++)
        {
            frames[i][j] *= window_func[j];
        }
    }

    // 5. fft
    int n = 256;
    for (i = 0; i < num_frames; i++)
    {
        for (j = 0; j < frame_length; j++)
        {
            fft_x[j] = frames[i][j];
        }

        rfft(fft_x, n);

        pow_frames[i][0] = (1.0 / n) * ((fft_x[0]) * (fft_x[0]));
        pow_frames[i][1] = (1.0 / n) * ((fft_x[1]) * (fft_x[1]) + (fft_x[n - 1]) * (fft_x[n - 1]));
        for (j = 2; j < n / 2; j += 2)
        {
            pow_frames[i][j] = (1.0 / n) * ((fft_x[j]) * (fft_x[j]) + (fft_x[n - j]) * (fft_x[n - j]));
            pow_frames[i][j + 1] = (1.0 / n) * ((fft_x[j + 1]) * (fft_x[j + 1]) + (fft_x[n - j - 1]) * (fft_x[n - j - 1]));
        }
        pow_frames[i][n / 2] = (1.0 / n) * ((fft_x[n / 2]) * (fft_x[n / 2]));

    }

    // 6. 三角滤波
    int low_freq_mel = 0;
    float delata = 0.f;
    int nfilt = 40;  // mel滤波器组：40个滤波器
    float high_freq_mel = 0.f;
    float mel_points[42] = { 0.f };
    float hz_points[42] = { 0.f };
    int bin[42] = { 0 };
    high_freq_mel = 2595 * log10f(6.714285);
	delata = (high_freq_mel - low_freq_mel) / (nfilt + 1);
	for (i = 0; i < 42; i++)
	{
		mel_points[i] = i * delata;
	}

	for (i = 0; i < 42; i++)
	{
		hz_points[i] = 700 * (powf(10, (mel_points[i] / 2595)) - 1);
	}

	for (i = 0; i < 42; i++)
	{
		bin[i] = floor((frame_length + 1) * hz_points[i] / RESAMPLE_WAV_LEN);
	}

	float f_m_minus = 0.f;
	float f_m = 0.f;
	float f_m_plus = 0.f;
	for (i = 1; i < 41; i++)
	{
		f_m_minus = bin[i - 1];
		f_m = bin[i];
		f_m_plus = bin[i + 1];
		for (j = f_m_minus; j < f_m; j++)
		{
			fbank[i - 1][j] = (float)(j - bin[i - 1]) / (float)(bin[i] - bin[i - 1]);
		}
		for (j = f_m; j < f_m_plus; j++)
		{
			fbank[i - 1][j] = (float)(bin[i + 1] - j) / (float)(bin[i + 1] - bin[i]);
		}

	}

	for (i = 0; i < num_frames; i++)
	{
		for (j = 0; j < 40; j++)
		{
			float sum = 0.f;
			for (k = 0; k < 129; k++)
			{
				sum += pow_frames[i][k] * fbank[j][k];
			}
			filter_banks[i][j] = sum;
		}
	}
	double eps = 1e-6; //10的-6次方
	for (i = 0; i < num_frames; i++)
	{
		for (j = 0; j < 40; j++)
		{
			filter_banks[i][j] = 20 * log10f(filter_banks[i][j] + eps);
		}
	}


	// 7. DCT
	float normlize = 0.2236068;
	for (i = 1; i < 13; i++)
	{
		for (j = 0; j < 40; j++)
		{
			dct_buff[i - 1][j] = normlize * cos(((double)M_2PI) / 2 / 40 * (j + 0.5) * i);
		}
	}

	for (i = 0; i < num_frames; i++)
	{
		for (j = 0; j < 12; j++)
		{
			float sum = 0.f;
			for (k = 0; k < 40; k++)
			{
				sum += filter_banks[i][k] * dct_buff[j][k];
			}
			mfcc[i][j] = sum;
		}
	}

	// 8. 正弦升降应用于MFCC以降低已被声称在噪声信号中改善语音识别的较高MFCC.
	float lift_buff[12] = { 0.f };
	uint8_t cep_lifter = 22;
	for (i = 0; i < 12; i++)
	{
		lift_buff[i] = 1 + ((float)cep_lifter / 2) * sin((double)M_2PI / 2 * i / 22);
	}

	for (i = 0; i < 50; i++)
	{
		for (j = 0; j < 12; j++)
		{
			mfcc[i][j] = mfcc[i][j] * lift_buff[j];
		}
	}

	// 9. 归一化操作
	for (i = 0; i < 50 * 12; i++)
	{
		if (*(mfcc_data + i) > a_max)
		{
			a_max = *(mfcc_data + i);
		}

		if (*(mfcc_data + i) < a_min)
		{
			a_min = *(mfcc_data + i);
		}
	}

	for (i = 0; i < 50 * 12; i++)
	{
		 *(mfcc_data + i) = (*(mfcc_data + i) - a_min) / (a_max - a_min);
	}


    printf(">>>>>");
}